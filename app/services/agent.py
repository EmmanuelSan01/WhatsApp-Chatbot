import logging
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from app.config import Config
from app.services.qdrant import QdrantService
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

# ==============================
# Agente RAG Principal
# ==============================

class AgentService:
    def __init__(self):
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        
        if Config.OPENAI_API_KEY:
            try:
                self.openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("✅ Cliente OpenAI inicializado para AgentService")
            except Exception as e:
                logger.error(f"Error inicializando OpenAI: {e}")
                self.openai_client = None
        else:
            logger.warning("⚠️ No se encontró OPENAI_API_KEY")
            self.openai_client = None

    async def process_query(self, query: str, user_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Procesa una consulta del usuario usando RAG (Qdrant + embeddings)
        """
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            query_embedding = await self.embedding_service.generate_embedding(query)
            logger.debug(f"Generated embedding with dimension: {len(query_embedding)}")

            relevant_docs = self.qdrant_service.search_similar(
                query_embedding,
                limit=5
            )
            
            if not relevant_docs:
                logger.warning("No relevant documents found in Qdrant - database may be empty")
                return {
                    "reply": "Lo siento, parece que la base de datos de cursos no está disponible en este momento. Por favor, contacta al administrador para sincronizar los datos.",
                    "sources": [],
                    "relevance_score": 0.0,
                    "context_used": [],
                    "error": "empty_database"
                }
            
            logger.info(f"Found {len(relevant_docs)} relevant documents")
            context_text = self._build_context(relevant_docs, context)

            response_text = await self._generate_response(query, context_text, user_id)

            return {
                "reply": response_text,
                "sources": [doc.get("payload", {}).get("nombre", "Unknown") for doc in relevant_docs],
                "relevance_score": relevant_docs[0].get("score", 0.0) if relevant_docs else 0.0,
                "context_used": [doc.get("payload", {}) for doc in relevant_docs]
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "reply": "Lo siento, hubo un error procesando tu consulta. Por favor intenta de nuevo o contacta al administrador si el problema persiste.",
                "sources": [],
                "relevance_score": 0.0,
                "context_used": [],
                "error": str(e)
            }

    def _build_context(self, relevant_docs: List[Dict], additional_context: Optional[Dict] = None) -> str:
        """
        Construye el contexto a partir de documentos relevantes de la KB
        """
        context_parts = []

        if relevant_docs:
            context_parts.append("� Información de cursos disponibles:")
            for doc in relevant_docs:
                payload = doc.get("payload", {})
                
                nombre = payload.get('nombre', 'N/A')
                descripcion = payload.get('descripcion', 'N/A')
                precio = payload.get('precio', 'N/A')
                categoria = payload.get('categoria', 'N/A')
                
                disponible = payload.get('disponible', False)
                
                estado_disponibilidad = "✅ Disponible" if disponible else "❌ No disponible"
                
                context_parts.append(f"- **{nombre}**")
                context_parts.append(f"  📝 Descripción: {descripcion}")
                context_parts.append(f"  💰 Precio: ${precio}")
                context_parts.append(f"  📂 Categoría: {categoria}")
                context_parts.append(f"  📦 Disponibilidad: {estado_disponibilidad}")
                
                promociones = payload.get('promociones_activas', '')
                if promociones and promociones.strip():
                    context_parts.append(f"  🎉 Promociones: {promociones}")
                
                context_parts.append("")

        if additional_context:
            context_parts.append("ℹ️ Información adicional:")
            for key, value in additional_context.items():
                context_parts.append(f"- {key}: {value}")

        return "\n".join(context_parts) if context_parts else "No se encontró información específica en la base de datos."

    async def _generate_response(self, query: str, context: str, user_id: str) -> str:
        """
        Genera respuesta usando OpenAI con el contexto de la KB real. Si falla, usa Google Generative AI (Gemini).
        """
        from app.config import Config
        import aiohttp

        system_prompt = """
    Eres HypatIA 🎓, asistente educativo especializado en cursos de Deep Learning y tecnologías afines.
        
        INSTRUCCIONES IMPORTANTES:
        - SOLO usa información del contexto proporcionado (datos reales de la base de datos)
        - La disponibilidad se indica claramente con ✅ Disponible o ❌ No disponible
    - Si un curso muestra ✅ Disponible, significa que ESTÁ DISPONIBLE para inscripción
    - Si un curso muestra ❌ No disponible, significa que NO ESTÁ DISPONIBLE para inscripción
        - Responde con precisión sobre la disponibilidad basándote únicamente en estos indicadores
        - La cantidad exacta de unidades no es relevante para el cliente
    - NO inventes precios, cursos o características que no estén en el contexto
        - Sé amigable, profesional y conciso
        - Incluye emojis relevantes para hacer la conversación más amena
        - Si el contexto está vacío, explica que necesitas más información
        
    Tu objetivo es ayudar a los estudiantes con información REAL y precisa sobre nuestros cursos.
        """

        user_prompt = f"""
        Consulta del cliente: {query}

        Información disponible en nuestra base de datos:
        {context}
        
        Por favor responde basándote ÚNICAMENTE en la información proporcionada arriba.
        Presta especial atención a los indicadores de disponibilidad (✅ Disponible / ❌ No disponible).
        """

        # Intentar con OpenAI
        try:
            if not self.openai_client:
                raise Exception("No OpenAI client")

            logger.debug("Sending request to OpenAI...")
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=600,
                temperature=0.3
            )
            generated_response = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI response generated successfully: {len(generated_response)} characters")
            return generated_response
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}")
            # Fallback a Google Generative AI (Gemini)
            if not Config.GOOGLE_API_KEY:
                return "Lo siento, hubo un error generando la respuesta y no hay API de Google configurada. Por favor intenta de nuevo o verifica la configuración."
            try:
                async with aiohttp.ClientSession() as session:
                    url = "https://generativelanguage.googleapis.com/v1beta/models/" + Config.GOOGLE_MODEL + ":generateContent?key=" + Config.GOOGLE_API_KEY
                    payload = {
                        "contents": [
                            {"parts": [{"text": user_prompt}]}
                        ]
                    }
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Gemini response parsing
                            try:
                                return data["candidates"][0]["content"]["parts"][0]["text"]
                            except Exception:
                                return "[Google Gemini] No se pudo extraer la respuesta del modelo."
                        else:
                            return f"[Google Gemini] Error en la generación: {resp.status}"
            except Exception as ge:
                logger.error(f"Error usando Google Gemini API: {str(ge)}")
                return "Lo siento, hubo un error generando la respuesta con Google Gemini API. Por favor intenta de nuevo o verifica la configuración."

# ==============================
# Agente Principal (Orquestador)
# ==============================

class HypatiaAgent:
    """
    Agente principal que usa únicamente la knowledge base real de Qdrant
    """
    def __init__(self):
        self.rag_agent = AgentService()

    async def process_message(self, message: str, user_info: Dict[str, Any] = None) -> str:
        """
        Procesa mensajes usando únicamente la knowledge base real
        """
        try:
            user_id = user_info.get("id", "anonimo") if user_info else "anonimo"
            response_dict = await self.rag_agent.process_query(message, user_id)
            response = response_dict.get("reply", "")
            
            if not response:
                return "Lo siento, no pude procesar tu consulta en este momento. Por favor intenta de nuevo."
                
            return response
            
        except Exception as e:
            logger.error(f"Error in HypatiaAgent.process_message: {str(e)}")
            return "Hubo un error procesando tu mensaje. Por favor intenta de nuevo."

    def get_model_info(self) -> Dict[str, Any]:
        """
        Información del modelo actual
        """
        return {
            "model": "HypatIA",
            "version": "1.0",
            "knowledge_source": "MySQL + Qdrant Vector DB",
            "description": "Asistente educativo con información real de cursos"
        }

    def is_available(self) -> bool:
        """
        Verifica si el agente está disponible
        """
        return self.rag_agent.openai_client is not None