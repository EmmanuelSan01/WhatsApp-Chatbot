"""
Agente principal que orquesta el sistema multi-agente
"""
import logging
import json
import hashlib
from typing import Dict, Any, Optional
from langroid import ChatAgent, ChatAgentConfig
from langroid.agent.tools import ForwardTool

from app.agents.config import langroid_config
from .specialized_agents import AnalyticsAgent
from .utils import safe_stringify

logger = logging.getLogger(__name__)

# Configuración explícita del logger para mostrar logs INFO
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class MainHypatiaAgent(ChatAgent):
    """Agente principal que orquesta el sistema multi-agente"""
    
    def __init__(self, config: ChatAgentConfig):
        super().__init__(config)
        
        # Solo configurar el agente de analíticas
        self.analytics_agent = AnalyticsAgent(
            ChatAgentConfig(
                llm=config.llm,
                system_message=langroid_config.SYSTEM_PROMPTS["analytics_agent"],
                name="AnalyticsAgent"
            )
        )
        
        # Herramientas habilitadas
        self.enable_message(ForwardTool)
        
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de conversación del analytics agent"""
        try:
            if hasattr(self, 'analytics_agent') and self.analytics_agent:
                return self.analytics_agent.get_metrics()
            else:
                # Retornar estadísticas por defecto si no hay analytics agent
                return {
                    "total_messages": 0,
                    "user_satisfaction": [],
                    "conversion_indicators": [],
                    "status": "analytics_agent_not_available"
                }
        except Exception as e:
            logger.error(f"Error getting conversation stats: {str(e)}")
            return {
                "total_messages": 0,
                "user_satisfaction": [],
                "conversion_indicators": [],
                "error": str(e)
            }


    async def handle_user_message(self, message: str, user_id: Optional[int] = None, 
                                  conversation_context: Optional[Dict] = None) -> str:
        """Maneja mensaje de usuario orquestando múltiples agentes, usando Redis para cacheo de resultados. Fallback a Google Gemini si falla OpenAI/Langroid."""
        import time
        start_time = time.time()
        from app.config import Config
        import aiohttp
        try:
            # Usar ServiceManager para obtener instancias singleton optimizadas
            from app.services.service_manager import service_manager
            redis_cache = service_manager.get_redis_cache()

            # Mejorar la clave de cache usando hash para evitar colisiones y asegurar unicidad
            cache_key = f"cursos:busqueda:{hashlib.sha256(message.strip().lower().encode()).hexdigest()}"
            cached_result = redis_cache.get(cache_key)
            if cached_result:
                logger.info(f"[CACHE HIT] Resultado recuperado desde Redis para clave: {cache_key}")
                knowledge_response = cached_result
            else:
                logger.info(f"[CACHE MISS] Generando nuevo resultado para clave: {cache_key}")
                # Ya no hay knowledge_agent, solo usar el mensaje original
                knowledge_response = message
                redis_cache.set(cache_key, knowledge_response, expire_seconds=600)  # Cache por 10 minutos

            self.analytics_agent.track_conversation(message, "")
            # Generar prompt simple solo con el mensaje del usuario
            context_prompt = f"Consulta del usuario: {message}\n\nPor favor responde de manera útil y profesional."
            try:
                final_response = await self.llm_response_async(context_prompt)
            except Exception as e:
                error_msg = str(e)
                # Fallback a Google Gemini si falla OpenAI/Langroid
                logger.error(f"Error in MainHypatiaAgent (OpenAI/Langroid): {error_msg}")
                if Config.GOOGLE_API_KEY:
                    try:
                        async with aiohttp.ClientSession() as session:
                            url = "https://generativelanguage.googleapis.com/v1beta/models/" + Config.GOOGLE_MODEL + ":generateContent?key=" + Config.GOOGLE_API_KEY
                            payload = {
                                "contents": [
                                    {"parts": [{"text": context_prompt}]}
                                ]
                            }
                            async with session.post(url, json=payload) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    try:
                                        gemini_response = data["candidates"][0]["content"]["parts"][0]["text"]
                                        self.analytics_agent.track_conversation(message, gemini_response)
                                        elapsed = time.time() - start_time
                                        logger.info(f"[RESPONSE TIME] El agente (Gemini) tardó {elapsed:.2f} segundos en generar la respuesta.")
                                        return gemini_response
                                    except Exception:
                                        return "[Google Gemini] No se pudo extraer la respuesta del modelo."
                                else:
                                    return f"[Google Gemini] Error en la generación: {resp.status}"
                    except Exception as ge:
                        logger.error(f"Error usando Google Gemini API: {str(ge)}")
                        return "Lo siento, hubo un error generando la respuesta con Google Gemini API. Por favor intenta de nuevo o verifica la configuración."
                else:
                    return "Lo siento, hubo un error procesando tu consulta y no hay API de Google configurada. Por favor intenta de nuevo."

                return "Lo siento, hubo un error procesando tu consulta. Por favor intenta de nuevo."

            self.analytics_agent.track_conversation(message, final_response)
            elapsed = time.time() - start_time
            logger.info(f"[RESPONSE TIME] El agente tardó {elapsed:.2f} segundos en generar la respuesta.")
            return final_response
        except Exception as e:
            logger.error(f"Error in MainHypatiaAgent: {str(e)}")
            return "Lo siento, hubo un error procesando tu consulta. Por favor intenta de nuevo."
