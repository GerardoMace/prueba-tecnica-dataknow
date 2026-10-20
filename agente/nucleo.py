"""Lógica principal del agente y conexión con la API de Gemini."""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date

from . import herramientas as H

INSTRUCCIONES = """Eres el agente analista de costos de una constructora. Ayudas a la gerencia a entender y usar
el modelo de costos de dos equipos críticos (Equipo1 y Equipo2) que dependen de tres materias primas (X, Y, Z).

Reglas:
- Responde en español, claro y breve, como consultor que habla con un gerente (no con un programador).
- Todo número sale de una herramienta. Nunca inventes cifras. Si no tienes el dato, dilo.
- Los precios históricos (incluido ago-2023) ya están dentro de las herramientas: nunca se los pidas al usuario.
- Los datos del modelo llegan hasta agosto de 2023 y el pronóstico cubre sep-2023 a ago-2024.
  Hoy es {hoy}. Si combinas con noticias actuales, aclara esa diferencia de fechas.
- Para combinar el modelo con conocimiento externo: busca noticias, extrae los precios o tendencias que
  mencionen (con fecha y fuente), y usa costo_con_precios / ubicar_en_pronostico / simular_escenario
  para traducirlos a costo de los equipos. Separa siempre "lo que dice el modelo" de "lo que dicen las noticias".
- X y Z se identificaron como petróleo Brent y aluminio comparándolos con series públicas (notebook 01, sección 4); Y es consistente con acero, pero su serie exacta no se identificó. Dilo así si es relevante.
- Si el usuario comparte un dato útil para después (plan de compras, cantidades, decisiones), guárdalo con guardar_nota.
- Cuando pidan un entregable (Excel, reporte), usa generar_reporte. Presupuesto con contingencia = P80 salvo que pidan otro nivel.
- Sé eficiente: pide varias herramientas a la vez cuando sean independientes y no repitas consultas.

Memoria de conversaciones anteriores:
{memoria}
"""


@dataclass
class Paso:
    """Registro de una herramienta usada."""
    herramienta: str
    argumentos: dict
    resultado: dict
    segundos: float


@dataclass
class Respuesta:
    texto: str
    pasos: list[Paso] = field(default_factory=list)
    modelo: str = ""


class LLM:
    """Interfaz mínima: dado el historial y las herramientas, regresa texto y/o llamadas a herramientas."""
    nombre = "base"

    def decidir(self, sistema: str, historial: list, herramientas: list[dict]):
        raise NotImplementedError

    def agregar_resultados(self, historial: list, crudo, resultados: list[tuple[str, dict]]):
        raise NotImplementedError

    def mensaje_usuario(self, texto: str):
        raise NotImplementedError

    def mensaje_modelo(self, texto: str):
        raise NotImplementedError


class GeminiLLM(LLM):
    """Gemini (Google AI Studio, cuenta gratuita). Se usa el modo de llamadas a funciones,
    pero con la ejecución automática apagada: el ciclo y la ejecución son nuestros."""

    def __init__(self, api_key: str, modelo: str = "gemini-flash-latest",
                 respaldo: str = "gemini-flash-lite-latest"):
        from google import genai
        from google.genai import types
        self.types = types
        self.cliente = genai.Client(api_key=api_key)
        self.nombre = self.usado = modelo
        self.respaldo = respaldo

    def _config(self, sistema, herramientas):
        t = self.types
        if not herramientas:
            return t.GenerateContentConfig(system_instruction=sistema, temperature=0.2)
        decl = [t.FunctionDeclaration(name=h["name"], description=h["description"],
                                      parameters_json_schema=h["parameters"]) for h in herramientas]
        return t.GenerateContentConfig(
            system_instruction=sistema,
            tools=[t.Tool(function_declarations=decl)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )

    def decidir(self, sistema, historial, herramientas):
        modelos = [self.nombre, self.respaldo]
        for intento in range(4):
            modelo = modelos[min(intento // 2, 1)]
            try:
                r = self.cliente.models.generate_content(
                    model=modelo, contents=historial, config=self._config(sistema, herramientas))
                self.usado = modelo
                break
            except Exception as e:
                if any(c in str(e) for c in ("429", "500", "503")) and intento < 3:
                    time.sleep(3 * (intento + 1))
                    continue
                raise
        contenido = r.candidates[0].content
        llamadas = [(p.function_call.name, dict(p.function_call.args or {}))
                    for p in contenido.parts if p.function_call]
        texto = "".join(p.text for p in contenido.parts if p.text and not getattr(p, "thought", False))
        return texto, llamadas, contenido

    def agregar_resultados(self, historial, crudo, resultados):
        t = self.types
        historial.append(crudo)
        historial.append(t.Content(role="user", parts=[
            t.Part.from_function_response(name=n, response={"resultado": r}) for n, r in resultados]))

    def mensaje_usuario(self, texto):
        return self.types.Content(role="user", parts=[self.types.Part.from_text(text=texto)])

    def mensaje_modelo(self, texto):
        return self.types.Content(role="model", parts=[self.types.Part.from_text(text=texto)])


class Agente:
    def __init__(self, llm: LLM, max_pasos: int = 8, max_turnos_memoria: int = 10):
        self.llm = llm
        self.max_pasos = max_pasos
        self.max_turnos = max_turnos_memoria
        self.historial = []

    def _sistema(self) -> str:
        notas = H.leer_memoria()
        memoria = "\n".join(f"- ({n['fecha']}) {n['nota']}" for n in notas[-20:]) or "(sin notas)"
        return INSTRUCCIONES.format(hoy=date.today().isoformat(), memoria=memoria)

    def preguntar(self, pregunta: str) -> Respuesta:
        # 1. PERCIBIR
        self.historial.append(self.llm.mensaje_usuario(pregunta))
        trabajo = list(self.historial)
        pasos: list[Paso] = []
        sistema = self._sistema()

        for _ in range(self.max_pasos):
            # 2. DECIDIR
            texto, llamadas, crudo = self.llm.decidir(sistema, trabajo, H.HERRAMIENTAS)
            if not llamadas:
                self._recordar(texto)
                return Respuesta(texto=texto, pasos=pasos, modelo=getattr(self.llm, "usado", self.llm.nombre))
            # 3. ACTUAR
            resultados = []
            for nombre, args in llamadas:
                t0 = time.time()
                res = H.ejecutar(nombre, args)
                pasos.append(Paso(nombre, args, res, round(time.time() - t0, 2)))
                resultados.append((nombre, res))
            # 4. OBSERVAR
            self.llm.agregar_resultados(trabajo, crudo, resultados)

        trabajo.append(self.llm.mensaje_usuario(
            "Ya no uses más herramientas. Responde ahora con la información obtenida."))
        texto, _, _ = self.llm.decidir(sistema, trabajo, [])
        self._recordar(texto)
        return Respuesta(texto=texto, pasos=pasos, modelo=getattr(self.llm, "usado", self.llm.nombre))

    def _recordar(self, texto: str):
        self.historial.append(self.llm.mensaje_modelo(texto))
        self.historial = self.historial[-2 * self.max_turnos:]

    def reiniciar(self):
        self.historial = []


def crear_agente():
    """Crea el agente con Gemini si hay API key; si no, regresa None (la app usa el modo sin IA)."""
    from dotenv import load_dotenv
    load_dotenv(H.RAIZ / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return Agente(GeminiLLM(key, os.getenv("GEMINI_MODEL", "gemini-flash-latest")))


def resultado_json(paso: Paso) -> str:
    return json.dumps(paso.resultado, ensure_ascii=False, indent=2)[:4000]
