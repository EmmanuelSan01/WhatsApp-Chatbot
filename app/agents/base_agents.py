"""
Implementación de agentes base usando Langroid Framework - Módulo de compatibilidad
Este archivo mantiene compatibilidad con importaciones existentes mientras delega al nuevo sistema modular.
"""

# Importar todo del nuevo sistema modular
from .main_agent import MainHypatiaAgent
from .factory import HypatiaAgentFactory
from .specialized_agents import KnowledgeAgent, SalesAgent, AnalyticsAgent
from .tools import CourseSearchTool, PromotionSearchTool, UserHistoryTool
from .utils import safe_stringify

# Re-exportar para compatibilidad con código existente
__all__ = [
    'MainHypatiaAgent',
    'HypatiaAgentFactory', 
    'KnowledgeAgent',
    'SalesAgent',
    'AnalyticsAgent',
    'CourseSearchTool',
    'PromotionSearchTool', 
    'UserHistoryTool',
    'safe_stringify'
]
