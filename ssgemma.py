from __future__ import annotations

import ast
import asyncio
import contextlib
import datetime
import json
import random
import signal
import sys
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import NoReturn

import aiofiles
import aiohttp
import typer
import uvloop
from loguru import logger
from pydantic import BaseModel, Field, PydanticJSONEncoder
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from thefuzz import fuzz

onsole = Console()
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

RICH_AVAILABLE = True
# Get the script's directory for relative paths
# Define script directory
SCRIPT_DIR = Path(__file__).parent.resolve()  # Use Path.parent and Path.resolve()

_rejected_sample_counter = 0

_rejected_sample_lock = asyncio.Lock()
_generated_sample_counter = 0  # NEW: Track total generated samples
_generated_sample_lock = asyncio.Lock()  # NEW: Lock for generated counter

DEFAULT_MODEL_ID = 'openai/gpt-oss-20b'  # CHANGED from 'Qwen/Qwen3-30B-A3B'
DEFAULT_TOP_P = 0.95  # NEW constant
DEFAULT_TEMPERATURE = 1
DEFAULT_COUNT = 100000000  # Keep DEFAULT_COUNT
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / 'enrichedwstreet_with_pii.jsonl'  # Use Path / operator
DEFAULT_POLLUTE = False  # TODO add pollutor
DEFAULT_POLLUTION_RATE = 1.0  # 100% pollution rate to ensure all samples are polluted
DEFAULT_REPEATED_PII = True
DEFAULT_OCR_SPACES = True
DEFAULT_OCR_CHARS = True
DEFAULT_ENCODING = True
DEFAULT_NUMBERS = True
DEFAULT_FORMATS = True
DEFAULT_SAMPLE_SOURCE = 'enrichedwstreet.jsonl'  # Use Path / operator
DEFAULT_NUM_WORKERS = 80
DEFAULT_BATCH_SIZE = 100

DEFAULT_TEMPERATURE = 1


# ===== PII Types and Constants =====
class PIIType(str, Enum):
    """Types of Personally Identifiable Information (PII)."""

    CEP = 'cep'  # "CEP"
    CITY_NAME = 'city_name'  # "Nome da cidade"
    CITY_UF = 'city_uf'  # "Sigla do estado/UF da cidade"
    STATE = 'state'  # "Nome do estado"
    PHONE = 'phone'  # "Número de telefone"
    STREET = 'street'  # "Nome da rua"
    BUILDING_NUMBER = 'building_number'  # "Número do prédio"
    NEIGHBORHOOD = 'neighborhood'  # "Bairro"
    NAME = 'name'  # "Primeiro nome"
    MIDDLE_NAME = 'middle_name'  # "Nome do meio"
    SURNAMES = 'surnames'  # "Sobrenome"
    CPF = 'cpf'  # "Número do CPF"
    RG = 'rg'  # "Número de RG"
    PIS = 'pis'  # "Número do PIS"
    CNPJ = 'cnpj'  # "Número do CNPJ"
    CEI = 'cei'  # "Número do CEI"
    BIRTHDATE = 'BIRTHDATE'  # "Data de nascimento"
    CREDITCARD = 'CREDITCARD'  # "Número do cartão de crédito"
    RACE_OR_ETHNICITY = 'RACE_OR_ETHNICITY'  # "Raça ou etnia"
    RELIGIOUS_CONVICTION = 'RELIGIOUS_CONVICTION'  # "Convicção religiosa"
    POLITICAL_OPINION = 'POLITICAL_OPINION'  # "Opinião política"
    ORGANIZATION_AFFILIATION = 'ORGANIZATION_AFFILIATION'  # "Filiação a organização"
    MEDICAL_DATA = 'MEDICAL_DATA'  # "Dados médicos"
    SEXUAL_DATA = 'SEXUAL_DATA'  # "Dados sexuais"
    EMAIL = 'email'  # "Endereço de e-mail"


# PII type descriptions (in Portuguese for generation)
PII_TYPES = {
    'building_number': 'building_number',
    'city_name': 'city_name',
    'city_uf': 'city_uf',
    'state': 'state',
    'phone': 'phone',
    'street': 'street',
    'neighborhood': 'neighborhood',
    'name': 'name',
    'middle_name': 'middle_name',
    'surnames': 'surnames',
    'cpf': 'cpf',
    'rg': 'rg',
    'pis': 'pis',
    'cnpj': 'cnpj',
    'cei': 'cei',
    'BIRTHDATE': 'BIRTHDATE',
    'CREDITCARD': 'CREDITCARD',
    'email': 'email',
    'cep': 'cep',
    'RACE_OR_ETHNICITY': 'RACE_OR_ETHNICITY',
    'RELIGIOUS_CONVICTION': 'RELIGIOUS_CONVICTION',
    'POLITICAL_OPINION': 'POLITICAL_OPINION',
    'ORGANIZATION_AFFILIATION': 'ORGANIZATION_AFFILIATION',
    'MEDICAL_DATA': 'MEDICAL_DATA',
    'SEXUAL_DATA': 'SEXUAL_DATA',
}


# English translations for PII masks
PII_MASK_TRANSLATIONS = {
    'building_number': 'Building Number',
    'city_name': 'city_name',
    'city_uf': 'city_name UF',
    'state': 'State',
    'CREDITCARD': 'Credit Card Number',
    'BIRTHDATE': 'Birth Date',
    'pis': 'Social Security Number (PIS)',
    'email': 'Email Address',
    'name': 'First Name',
    'middle_name': 'Middle Name',
    'surnames': 'Last Name',
    'cpf': 'Social Security Number (CPF)',
    'street': 'Street Name',
    'phone': 'Phone Number',
    'cep': 'Zip Code',
    'neighborhood': 'Neighborhood',
    'rg': 'ID Number (RG)',
    'RACE_OR_ETHNICITY': 'Race or Ethnicity',
    'RELIGIOUS_CONVICTION': 'Religious Conviction',
    'POLITICAL_OPINION': 'Political Opinion',
    'ORGANIZATION_AFFILIATION': 'Organization Affiliation',
    'MEDICAL_DATA': 'Medical Data',
    'SEXUAL_DATA': 'Sexual Data',
    'cnpj': 'Company Registration Number',
    'cei': 'Employer Registration Number',
}

# Updated weighted probabilities for PII types (reflecting real-world frequency)
PII_TYPE_WEIGHTS = {
    'name': 0.7,
    'surnames': 0.7,
    'email': 0.01,
    'phone': 0.05,
    'city_name': 0.03,
    'city_uf': 0.02,
    'cep': 0.02,
    'street': 0.05,
    'building_number': 0.01,
    'CREDITCARD': 0.005,
    'BIRTHDATE': 0.03,
    'cpf': 0.04,
    'rg': 0.05,
    'neighborhood': 0.03,
    'middle_name': 0.02,
    'state': 0.02,
    'pis': 0.05,
    'RACE_OR_ETHNICITY': 0.07,
    'RELIGIOUS_CONVICTION': 0.07,
    'POLITICAL_OPINION': 0.07,
    'ORGANIZATION_AFFILIATION': 0.07,
    'MEDICAL_DATA': 0.07,
    'SEXUAL_DATA': 0.07,
}


# Define PII categories for context/intent biasing
PII_CATEGORIES = {
    'medical': {'MEDICAL_DATA', 'BIRTHDATE', 'ID_RG', 'name', 'LAST_NAME', 'SEXUAL_DATA', 'PHONE'},
    'financial': {
        'SSN_CPF',
        'ID_RG',
        'PHONE',
        'email',
        'street',
        'building_number',
        'NEIGHBORHOOD',
        'city_name',
        'STATE',
        'city_uf',
        'ZIPCODE_CEP',
        'CREDITCARD',
        'SOCIAL_NB_PIS',
        'name',
        'LAST_NAME',
    },
    'employment': {
        'name',
        'LAST_NAME',
        'email',
        'PHONE',
        'BIRTHDATE',
        'ORGANIZATION_AFFILIATION',
        'ID_RG',
        'SSN_CPF',
        'SOCIAL_NB_PIS',
        'street',
        'building_number',
        'NEIGHBORHOOD',
        'city_name',
        'STATE',
        'city_uf',
        'ZIPCODE_CEP',
    },
    'legal': {
        'SSN_CPF',
        'ID_RG',
        'name',
        'LAST_NAME',
        'BIRTHDATE',
        'street',
        'building_number',
        'NEIGHBORHOOD',
        'city_name',
        'STATE',
        'city_uf',
        'ZIPCODE_CEP',
        'POLITICAL_OPINION',
    },
    'educational': {
        'name',
        'LAST_NAME',
        'BIRTHDATE',
        'ID_RG',
        'SSN_CPF',
        'email',
        'PHONE',
        'street',
        'building_number',
        'NEIGHBORHOOD',
        'city_name',
        'STATE',
        'city_uf',
        'ZIPCODE_CEP',
    },
    'personal_id': {'name', 'middle_name', 'LAST_NAME', 'BIRTHDATE', 'ID_RG', 'SSN_CPF', 'PHONE', 'email'},
    'address': {'street', 'building_number', 'NEIGHBORHOOD', 'city_name', 'STATE', 'city_uf', 'ZIPCODE_CEP'},
    'social_event': {'name', 'LAST_NAME', 'PHONE', 'email', 'city_name', 'NEIGHBORHOOD'},
    'sensitive_personal': {
        'SEXUAL_DATA',
        'RELIGIOUS_CONVICTION',
        'POLITICAL_OPINION',
        'RACE_OR_ETHNICITY',
        'MEDICAL_DATA',
    },
}


STYLES = {
    'formal': {
        'formal1': 'Estruturado e profissional, adequado para contextos oficiais.',
        'formal2': 'Polido e respeitoso, aderindo a convenções formais.',
        'formal3': 'Claro e autoritativo, usado para comunicação profissional.',
        'formal4': 'Impessoal e preciso, evitando linguagem casual.',
        'formal5': 'Sério e apropriado, frequentemente encontrado em escrita jurídica ou acadêmica.',
        'tecnico1': 'Detalhado e preciso, explicando conceitos claramente.',
        'tecnico2': 'Rico em jargões e factual, adaptado para especialistas.',
        'tecnico3': 'Objetivo e estruturado, apresentando dados ou instruções.',
        'tecnico4': 'Focado e metódico, projetado para resolução de problemas.',
        'tecnico5': 'Conciso e sistemático, entregando detalhes técnicos.',
        'tecnico': 'Preciso e factual, usado para explicações científicas ou detalhadas.',
        'persuasivo': 'Convincente e influente, projetado para mudar opiniões ou ações.',
        'descritivo': 'Vívido e detalhado, criando uma imagem mental clara.',
        'instrucional': 'Claro e passo a passo, visando orientar ou ensinar.',
        'jornalistico': 'Objetivo e conciso, focando em fatos e eventos atuais.',
        'retorico': 'Argumentativo e impactante, projetado para provocar pensamento.',
        'prontuario_medico': 'Estilo de escrita tipico de prontuarios medicos, com abreviacoes e termos tecnicos.',
        'juridico': 'Linguagem formal e tecnica usada em documentos judiciais brasileiros.',
        'academico': 'Estilo formal, objetivo e estruturado usado em trabalhos científicos e acadêmicos.',
        'comercial': 'Linguagem persuasiva e direta utilizada em comunicações comerciais e marketing.',
        'telemarketing': 'Estilo de script utilizado em ligações de telemarketing, direto e protocolar.',
        'burocratico': 'Extremamente formal e padronizado, típico de documentos governamentais.',
        'simplificado': 'Linguagem muito simples e direta, adequada para comunicação com público amplo.',
    },
    'informal': {
        'casual': 'Relaxado e conversacional, como conversar com um amigo.',
        'narrativo': 'Semelhante a uma história, focado em eventos ou personagens em sequência.',
        'humoristico': 'Brincalhão e espirituoso, visando entreter ou divertir.',
        'emocional': 'Expressivo e sincero, focado em sentimentos ou visões pessoais.',
        'cartas': 'Cartas pessoais entre amigos ou entre familiares ou oficiais.',
        'erros_comuns': 'Texto com erros comuns de digitacao e gramatica.',
        'informal_internet': 'Estilo de escrita típico de redes sociais, com abreviações e gírias da internet.',
        'dialect_nordestino': 'Usando expressões e construções típicas do dialeto nordestino brasileiro.',
        'dialect_sulista': 'Incorporando expressões e construções típicas do sul do Brasil.',
        'mesclado_estrangeirismos': 'Texto com inclusão frequente de termos em inglês comuns no cotidiano brasileiro.',
        'registro_rapido': 'Notas rápidas como se fossem anotadas em pressa, com abreviações próprias.',
        'frases_curtas_fragmentadas': 'Estilo com frases muito curtas ou fragmentadas',
        'girias_internet_especificas': 'Uso de gírias específicas da internet (e.g., memes, siglas)',
        'portunhol': 'Mistura de português e espanhol',
        'transcricao_voz_texto_erros': 'Estilo de transcrição de voz para texto, com erros típicos',
    },
}

# --- CONTEXTS Dictionary (Updated) ---
CONTEXTS = {
    # --- Formal ---
    'formal': {
        'atendimento_bancario': 'Durante atendimento em agência bancária',
        'ambiente_juridico': 'Processos, contratos, petições e outros pedidos.',
        'reuniao_trabalho': 'Participando de reunião profissional ou empresarial',
        'registro_cartorio': 'Em procedimento cartorial',
        'audiencia_judicial': 'Durante audiência em tribunal ou vara judicial',
        'debate_formal': 'Participando de debate formal estruturado',
        'documentacao_rh': 'Formulários e documentos de departamento de Recursos Humanos',
        'declaracao_imposto': 'Preenchimento da declaração anual de imposto de renda',
        'certidao_civil': 'Solicitação de certidão em cartório civil',
        'consulta_medica': 'Durante consulta médica',
        'prontuario_medico': 'Preenchendo prontuário médico',
        'cadastro_saude': 'Preenchendo formulário de plano de saúde ou seguro de vida',
        'emergencia_hospital': 'Atendimento em pronto-socorro ou emergência hospitalar',
        'consultorio_medico': 'Durante atendimento em consultório médico particular',
        'notas_residente': 'Anotações feitas por médicos residentes sobre pacientes',
        'receita_medica': 'Preenchimento de receituário médico ou pedido de exames',
        'teleconsulta': 'Consulta médica realizada remotamente',
        'seguro_vida': 'Contratação de seguro de vida ou residencial',
        'arquivo_medico': 'Registro em arquivo médico físico ou digital',
        'entrevista_emprego': 'Em processo de entrevista de emprego',
        'processo_seletivo': 'Documentação para processo seletivo acadêmico',
        'registro_profissional': 'Cadastro em conselho profissional como OAB, CRM, CREA',
        'boletim_ocorrencia': 'Registro de boletim de ocorrência policial',
        'testamento': 'Elaboração de testamento ou documento de sucessão',
        'imigracao': 'Documentação para processos de imigração ou visto',
        'concurso_publico': 'Processo de seleção para cargo público',
        'ida_inss': 'Atendimento em agência do INSS',
        'documento_rh': 'Formulários e documentos de departamento de Recursos Humanos',
        'compra_imovel': 'Processo de compra e financiamento de imóvel',
        'servico_militar': 'Conversa sobre alistamento ou serviço militar',
        'exame_admissorio': 'Realizando exame médico admissional',
        'locacao_imovel': 'Processo de locação de imóvel',
        'abertura_conta': 'Abrindo nova conta bancária',
        'cadastro_governo': 'Cadastro em sistema governamental',
        'solicitacao_credito': 'Pedido de empréstimo ou financiamento',
        'cadastro_investimento': 'Abrindo conta em corretora de valores',
        'assinatura_servico': 'Contratando serviços por assinatura',
        'checkin_hotel': 'Fazendo check-in em hotel',
        'compra_presencial': 'Realizando compra em loja física com cadastro de cliente',
        'matricula_escolar': 'Realizando matrícula em instituição de ensino',
        'universidade': 'Contexto universitário e acadêmico',
        'transferencia_escolar': 'Processo de transferência entre instituições de ensino',
        'programa_social': 'Cadastro em programa social governamental',
        'reuniao_maconaria': 'Reunião de maçonaria (contexto social)',
        'reuniao_demolay': 'Reunião da Ordem DeMolay (contexto social)',
        'reuniao_clube_academico': 'Reunião de clube acadêmico/universitário',
        'cadastro_online': 'Preenchendo formulário de cadastro online',
        'suporte_tecnico': 'Em chamada com suporte técnico',
        'compra_online': 'Realizando compra em e-commerce',
        'discussao_politica': 'Em debate ou discussão sobre temas políticos',
        'escola_publica': 'Ambiente de escola pública brasileira',
        'sessao_terapia': 'Durante sessão de terapia',
    },
    # --- Informal ---
    'informal': {
        'cadastro_online': 'Preenchendo formulário de cadastro online',
        'suporte_tecnico': 'Em chamada com suporte técnico',
        'compra_online': 'Realizando compra em e-commerce',
        'encontro_amigos': 'Conversação informal entre amigos',
        'pai_filho': 'Comunicação entre pai e filho',
        'pai_filha': 'Comunicação entre pai e filha',
        'mae_filho': 'Comunicação entre mãe e filho',
        'mae_filha': 'Comunicação entre mãe e filha',
        'avos_netos': 'Interação entre avós e netos',
        'fila_banco': 'Esperando em fila de banco',
        'interior_brasil': 'Em cidade pequena do interior do Brasil',
        'metropole_brasil': 'Em grande centro urbano brasileiro',
        'supermercado': 'Cadastro de cliente em rede de supermercados',
        'discussao_politica': 'Em debate ou discussão sobre temas políticos',
        'escola_publica': 'Ambiente de escola pública brasileira',
        'feira_agropecuaria': 'Feira agropecuária',
        'recadinho_escola': 'Recado escolar informal (agenda, bilhete)',
        'partida_futebol': 'Assistindo ou comentando partida de futebol',
        'ida_praia': 'Conversa na praia ou planejando ir',
        'festa_junina': 'Durante festa junina',
        'feira_domingo': 'Passeio na feira de domingo',
        'onibus_trabalho': 'Conversa no ônibus indo para o trabalho',
        'trem_casa': 'Conversa no trem voltando para casa',
        'transito': 'Preso no trânsito, comentando situação',
        'comentarios_rede_social': 'Seção de comentários em rede social',
        'dm_rede_social': 'Mensagem direta (DM) em rede social',
        'chat_jogo_online': 'Chat durante jogo online',
        'discussao_forum': 'Discussão em fórum online',
        'blog_pessoal_comentarios': 'Blog pessoal ou seção de comentários',
        'reuniao_clube_hobby': 'Reunião informal de clube ou hobby',
        'conversa_vizinhos': 'Conversa casual com vizinhos',
        'jogo_buraco': 'Durante jogo de cartas (buraco)',
        'festa_reveillon': 'Festa de Ano Novo (Réveillon)',
        'pascoa': 'Celebração ou conversa sobre Páscoa',
        'ceia_natal': 'Durante a ceia de Natal',
        'ultimo_dia_aula': 'Conversa no último dia de aula',
        'avaliacao_cliente_texto': 'Escrevendo uma avaliação de cliente/produto',
        'msg_planejando_evento': 'Mensagens de texto planejando evento',
        'msg_app_namoro': 'Conversa em aplicativo de namoro',
        'fila_sus': 'Esperando atendimento no SUS',
        'hora_jornal': 'Comentando notícias durante o jornal na TV',
        'hora_novela': 'Comentando a novela na TV',
        'festa_aniversario': 'Durante festa de aniversário',
        '100_anos_avo': 'Comemoração de 100 anos do avô/avó',
        'festa_santa_rita': 'Festa de Santa Rita (religiosa/local)',
        'show_pirotecnico': 'Assistindo a show de fogos de artifício',
        'bate_boca': 'Discussão acalorada / Bate-boca',
        'funeral': 'Conversa durante um funeral ou velório',
        'parada_gay': 'Parada do Orgulho LGBTQ+',
        'quinta_carnaval': 'Conversa na quinta-feira antes do carnaval',
        'quarta_cinzas': 'Conversa na Quarta-feira de Cinzas',
        'dia_aviacao': 'Comentários sobre o Dia da Aviação/Santos Dumont',
        'dia_tiradentes': 'Comentários sobre o feriado de Tiradentes',
        'chegada_corte_brasil': 'Comentários sobre a chegada da corte portuguesa',
        'revolta_males': 'Comentários sobre a Revolta dos Malês',
        'recitacao_romance_jandira': 'Recitação ou discussão do Romance de Jandira',
        'cadastro_app': 'Criando perfil em aplicativo de serviços',
        'sessao_terapia': 'Durante sessão de terapia',
    },
}


# --- MEDIUMS Dictionary (Updated) ---
MEDIUMS = {
    'formal': {
        'email': 'Email corporativo ou pessoal',  # Can be formal or informal
        'documento': 'Documento técnico ou relatório',
        'carta_formal': 'Correspondência formal impressa',
        'contrato': 'Documento contratual entre partes',
        'declaracao': 'Declaração formal para fins específicos',
        'procuracao': 'Procuração para representação legal',
        'relatorio_interno': 'Relatório interno de empresa',
        'processo_judicial': 'Autos de processo judicial',
        'ata_audiencia': 'Ata ou registro formal de audiência',
        'documento_debate': 'Anotações ou transcrição de debate formal',
        'requerimento_inss': 'Formulário de solicitação ao INSS',
        'documentacao_rh': 'Pasta de documentos de funcionário em RH',
        'curriculum': 'Currículo profissional',
        'ficha_cadastral': 'Ficha de cadastro manual ou física',
        'prontuario': 'Prontuário físico ou eletrônico',
        'receita_medica': 'Prescrição médica física ou digital',
        'nota_fiscal': 'Nota fiscal eletrônica ou física',
        'intranet': 'Postagem em ambiente de intranet corporativa',
        'formulario_saude': 'Formulário de histórico médico',
        'prontuario_residencia': 'Notas de evolução de médico residente',
        'laudo_emergencia': 'Documento de atendimento emergencial',
        'historico_academico': 'Histórico escolar ou acadêmico',
        'boletim_escolar': 'Boletim ou relatório de desempenho escolar',
        'prova_identidade': 'Cópia de documento de identidade para comprovação',
        'arquivo_medico': 'Registro em arquivo médico físico ou digital',
        'solicitacao_emprego': 'Carta ou formulário de solicitação de emprego',
        'formulario': 'Formulário de cadastro (geral)',  # Can be formal
        'anotacao': 'Anotação pessoal ou profissional',  # Can be informal
        'grupo_discussao': 'Mensagens em grupo de discussão política',  # Can be informal
        'jornal_interior': 'Publicação em jornal de cidade do interior',  # Can be informal
        'anuncio_classificado': 'Anúncio em classificados de emprego',  # Can be informal
        'formulario_escola': 'Ficha cadastral de aluno em escola pública',  # Can be informal
        'senha_atendimento': 'Comprovante de senha para atendimento em instituição',  # Context dependent
    },
    'informal': {
        'email': 'Email corporativo ou pessoal',  # Can be formal or informal
        'mensagem': 'Mensagem de texto ou WhatsApp',
        'rede_social': 'Post ou comentário em rede social (geral)',
        'anotacao': 'Anotação pessoal ou profissional',  # Can be informal
        'chat_atendimento': 'Chat de atendimento ao cliente',
        'audio_gravado': 'Transcrição de áudio de ligação gravada',  # Often informal speech
        'chat_bot': 'Interação com assistente virtual automatizado',
        'caderneta_anotacoes': 'Caderneta ou livro de anotações pessoais manuscritas',
        'grupo_discussao': 'Mensagens em grupo de discussão política',  # Can be informal
        'bilhete_pessoal': 'Bilhete ou recado manuscrito entre pessoas próximas',
        'carta_familiar': 'Correspondência informal entre familiares',
        'fidelidade_mercado': 'Cadastro em programa de fidelidade de supermercado',  # Often informal
        'senha_atendimento': 'Comprovante de senha para atendimento em instituição',  # Context dependent
        'jornal_interior': 'Publicação em jornal de cidade do interior',  # Can be informal
        'anuncio_classificado': 'Anúncio em classificados de emprego',  # Can be informal
        'formulario_escola': 'Ficha cadastral de aluno em escola pública',  # Can be informal
        'databases': 'Databases e datasets que se tornaram públicos por acidente.',  # Often informal leakage
        'post_forum': 'Postagem em fórum online',
        'comentario_blog': 'Comentário em blog',
        'log_chat_jogo': 'Log de chat de jogo online',
        'legenda_story_rede_social': 'Legenda ou texto em story de rede social',
        'texto_avaliacao_cliente': 'Texto de avaliação de cliente/produto',
        'log_msg_app_namoro': 'Log de mensagens de aplicativo de namoro',
        'entrada_diario_pessoal': 'Entrada em diário pessoal (físico/digital)',
        'transcricao_sessao_terapia': 'Transcrição de sessão de terapia',
        'transcricao_bate_boca': 'Transcrição de discussão/bate-boca',
        'audio_whatsapp': 'Transcrição de áudio de WhatsApp',
    },
}

# --- INTENTS Dictionary (Updated) ---
INTENTS = {
    'formal': {
        'informar': 'Fornecer informação ou dados ao leitor',
        'solicitar': 'Pedir informação ou ação do leitor',
        'confirmar': 'Confirmar dados ou informações',
        'atualizar': 'Atualizar registros ou cadastros',
        'autorizar': 'Dar permissão ou autorização',
        'cadastrar': 'Realizar novo cadastro',
        'publicar': 'Publicar em diário oficial fatos, decisões ou resultados',
        'anunciar': 'Anunciar decisão ou fato',
        'formalizar': 'Formalizar ao público determinada informação ou decisão',
        'memoria_oficial': 'Documentos oficiais, públicos e privados que guardem a memória oficial de instituições (e.g., memorandos) ou profissionais (e.g., notas médicas)',
        'tornar_oficial': 'Escritos que memorializam determinados fatos, relacionados a algum pedido oficial (ex.: descrição de fatos de uma sentença)',
        'controle_interno': 'Dados guardados por profissionais para controle interno (ex.: dados médicos)',
        'identificar': 'Confirmar a identidade de uma pessoa',
        'verificar': 'Validar a veracidade de informações fornecidas',
        'notificar': 'Comunicar oficialmente sobre situação ou decisão',
        'comprovar': 'Apresentar evidência de fatos ou qualificações',
        'rastrear': 'Documentar histórico ou sequência de eventos',
        'autenticar': 'Verificar autenticidade de documentos ou identidade',
        'processar': 'Dar encaminhamento a uma solicitação ou pedido',
        'arquivar': 'Guardar informações para referência futura',
        'documentar': 'Registrar formalmente um evento ou transação',
        'consultar': 'Recuperar informações previamente registradas',
        'orientar': 'Fornecer direcionamento com base em dados pessoais',
        'diagnostico': 'Avaliar condição médica com base em informações pessoais',
        'contactar': 'Estabelecer meio de comunicação com a pessoa',
        'cobrar': 'Solicitar pagamento de valor devido',
        'aprovar': 'Conceder autorização com base em critérios pessoais',
        'registrar_emergencia': 'Documentar atendimento emergencial ou urgente',
        'testemunhar': 'Registrar depoimento ou testemunho formal',
        'educar': 'Fornecer informação educativa personalizada',
        'treinar': 'Documentação para treinamento de pessoal',
        'selecionar': 'Avaliar candidato para processo seletivo',
        'classificar': 'Categorizar pessoa por critérios específicos',
        'homologar': 'Validar oficialmente um documento ou situação',
        'agendar': 'Marcar compromisso ou atendimento',
        'prescrever': 'Receitar medicamento ou tratamento',
        'encaminhar': 'Direcionar pessoa para outro serviço ou profissional',
        'certificar': 'Emitir comprovação oficial de qualificação',
        'convocar': 'Chamar oficialmente para compromisso ou obrigação',
        'declarar': 'Fazer afirmação formal sobre situação ou status',
        'advertir': 'Aplicar advertência formal ou notificação',
        'habilitar': 'Conceder permissão para exercício de atividade',
        'inscrever': 'Incluir em lista, programa ou benefício',
        'qualificar': 'Avaliar aptidão ou qualificação para função',
        'avaliar_risco': 'Determinar níveis de risco em situação específica',
        'acompanhar': 'Monitorar evolução de caso ou situação',
        'alertar': 'Avisar sobre riscos ou problemas',  # Can be both formal and informal
        'reclamar': 'Registrar reclamação ou problema',  # Can be both formal and informal
    },
    'informal': {
        'alertar': 'Avisar sobre riscos ou problemas',  # Can be both formal and informal
        'reclamar': 'Registrar reclamação ou problema',  # Can be both formal and informal
        'memoria_pessoal': 'Notas pessoais que as pessoas usam para não esquecer de certos fatos',
        'compartilhar_intimidade': 'Comunicação íntima entre pessoas próximas',
        'pedir_conselho_informal': 'Pedir conselho de forma informal',
        'desabafar_reclamar': 'Desabafar ou reclamar sobre algo',
        'compartilhar_noticia_casual': 'Compartilhar notícias de forma casual',
        'fazer_planos_combinar': 'Fazer planos ou combinar algo',
        'brincar_zoar': 'Brincar, zoar ou fazer piada',
        'expressar_opiniao_casual': 'Expressar opinião de forma casual',
        'pedir_ajuda_naourgente': 'Pedir ajuda para algo não urgente',
        'relembrar_contar_causo': 'Relembrar ou contar uma história/causo',
        'registrar_funeral': 'Registrar condolências ou informações sobre funeral',
    },
}


# Define mappings from context/intent keys to PII categories (Updated)
CONTEXT_INTENT_TO_CATEGORY = {
    # Medical Contexts/Intents
    'consulta_medica': 'official',
    'prontuario_medico': 'official',
    'cadastro_saude': 'official',
    'emergencia_hospital': 'official',
    'consultorio_medico': 'official',
    'notas_residente': 'official',
    'receita_medica': 'official',
    'teleconsulta': 'official',
    'seguro_vida': 'official',
    'arquivo_medico': 'official',
    'diagnostico': 'official',
    'prescrever': 'official',
    'registrar_emergencia': 'official',
    'acompanhar': 'official',
    'avaliar_risco': 'official',
    'fila_sus': 'official',
    'exame_admissorio': 'employment',
    'sessao_terapia': 'private_life',
    'transcricao_sessao_terapia': 'private_life',
    # Financial/Commercial Contexts/Intents
    'atendimento_bancario': 'official',
    'cadastro_online': 'official',
    'compra_online': 'official',
    'checkin_hotel': 'official',
    'locacao_imovel': 'official',
    'abertura_conta': 'official',
    'cadastro_governo': 'official',
    'compra_presencial': 'official',
    'cadastro_app': 'official',
    'solicitacao_credito': 'official',
    'cadastro_investimento': 'official',
    'assinatura_servico': 'official',
    'supermercado': 'public_life',
    'fila_banco': 'public_life',
    'compra_imovel': 'official',
    'declaracao_imposto': 'official',
    'programa_social': 'official',
    'cadastrar': 'official',
    'identificar': 'official',
    'autenticar': 'official',
    'verificar': 'official',
    'atualizar': 'official',
    'autorizar': 'official',
    'processar': 'official',
    'cobrar': 'official',
    'aprovar': 'official',
    'inscrever': 'official',
    'avaliacao_cliente_texto': 'public_life',
    'texto_avaliacao_cliente': 'public_life',
    'fidelidade_mercado': 'public_life',
    # Employment Contexts/Intents
    'entrevista_emprego': 'employment',
    'processo_seletivo': 'employment',
    'documento_rh': 'employment',
    'reuniao_trabalho': 'employment',
    'registro_profissional': 'employment',
    'primeiro_emprego': 'employment',
    'solicitacao_emprego': 'employment',
    'selecionar': 'employment',
    'comprovar': 'employment',
    'qualificar': 'employment',
    'habilitar': 'employment',
    'treinar': 'employment',
    'momento_demissao': 'employment',
    'documentacao_rh': 'employment',
    'anuncio_classificado': 'employment',
    # Legal Contexts/Intents
    'registro_cartorio': 'official',
    'ambiente_juridico': 'official',
    'boletim_ocorrencia': 'official',
    'testamento': 'official',
    'imigracao': 'official',
    'audiencia_judicial': 'official',
    'concurso_publico': 'official',
    'certidao_civil': 'official',
    'formalizar': 'official',
    'memoria_oficial': 'official',
    'tornar_oficial': 'official',
    'testemunhar': 'official',
    'homologar': 'official',
    'certificar': 'official',
    'convocar': 'official',
    'declarar': 'official',
    'advertir': 'official',
    'servico_militar': 'official',
    'burocratico': 'official',
    # Educational Contexts/Intents
    'matricula_escolar': 'official',
    'inscricao_concurso': 'official',
    'escola_publica': 'public_life',
    'universidade': 'public_life',
    'transferencia_escolar': 'official',
    'educar': 'public_life',
    'recadinho_escola': 'private_life',
    'ultimo_dia_aula': 'public_life',
    'reuniao_clube_academico': 'public_life',
    'formulario_escola': 'official',
    'academico': 'public_life',
    # Social/Personal/Informal Contexts/Intents
    'encontro_amigos': 'private_life',
    'pai_filho': 'private_life',
    'pai_filha': 'private_life',
    'mae_filho': 'private_life',
    'mae_filha': 'private_life',
    'avos_netos': 'private_life',
    'interior_brasil': 'public_life',
    'metropole_brasil': 'public_life',
    'feira_agropecuaria': 'public_life',
    'partida_futebol': 'public_life',
    'ida_praia': 'private_life',
    'festa_junina': 'public_life',
    'feira_domingo': 'public_life',
    'onibus_trabalho': 'public_life',
    'trem_casa': 'public_life',
    'transito': 'public_life',
    'comentarios_rede_social': 'public_life',
    'dm_rede_social': 'private_life',
    'chat_jogo_online': 'public_life',
    'discussao_forum': 'public_life',
    'blog_pessoal_comentarios': 'public_life',
    'reuniao_clube_hobby': 'public_life',
    'conversa_vizinhos': 'private_life',
    'jogo_buraco': 'private_life',
    'festa_reveillon': 'public_life',
    'pascoa': 'private_life',
    'ceia_natal': 'private_life',
    'msg_planejando_evento': 'private_life',
    'msg_app_namoro': 'intimacy',
    'log_msg_app_namoro': 'intimacy',
    'hora_jornal': 'private_life',
    'hora_novela': 'private_life',
    'festa_aniversario': 'private_life',
    '100_anos_avo': 'private_life',
    'festa_santa_rita': 'public_life',
    'show_pirotecnico': 'public_life',
    'bate_boca': 'intimacy',
    'transcricao_bate_boca': 'intimacy',
    'funeral': 'private_life',
    'parada_gay': 'public_life',
    'reuniao_maconaria': 'employment',
    'reuniao_demolay': 'employment',
    'quinta_carnaval': 'public_life',
    'quarta_cinzas': 'public_life',
    'dia_aviacao': 'public_life',
    'dia_tiradentes': 'public_life',
    'chegada_corte_brasil': 'public_life',
    'revolta_males': 'public_life',
    'recitacao_romance_jandira': 'public_life',
    'pedir_conselho_informal': 'private_life',
    'desabafar_reclamar': 'private_life',
    'compartilhar_noticia_casual': 'private_life',
    'fazer_planos_combinar': 'private_life',
    'brincar_zoar': 'private_life',
    'expressar_opiniao_casual': 'private_life',
    'pedir_ajuda_naourgente': 'private_life',
    'relembrar_contar_causo': 'private_life',
    'registrar_funeral': 'private_life',
    'alertar': 'public_life',
    'reclamar': 'public_life',
    'memoria_pessoal': 'private_life',
    'compartilhar_intimidade': 'intimacy',
    'suporte_tecnico': 'public_life',
    'post_forum': 'public_life',
    'comentario_blog': 'public_life',
    'log_chat_jogo': 'public_life',
    'legenda_story_rede_social': 'public_life',
    'entrada_diario_pessoal': 'intimacy',
    'audio_whatsapp': 'private_life',
    'grupo_discussao': 'public_life',
    'bilhete_pessoal': 'private_life',
    'carta_familiar': 'private_life',
    'senha_atendimento': 'official',
    'jornal_interior': 'public_life',
    'databases': 'official',
    'informal_internet': 'public_life',
    'dialect_nordestino': 'public_life',
    'dialect_sulista': 'public_life',
    'mesclado_estrangeirismos': 'public_life',
    'simplificado': 'public_life',
    'registro_rapido': 'official',
    'frases_curtas_fragmentadas': 'public_life',
    'girias_internet_especificas': 'public_life',
    'portunhol': 'public_life',
    'transcricao_voz_texto_erros': 'public_life',
}


# Initialize Rich console for pretty output


# Initialize Typer app
app = typer.Typer(help='Clean PII data generator', add_completion=False, rich_markup_mode='rich')

# Define the script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()  # Use Path.parent and Path.resolve()

# Configure logging
logger.remove(0)  # Remove the default handler
logger.add(
    RichHandler(rich_tracebacks=True, markup=True, show_path=False, tracebacks_show_locals=True),
    format=(
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
    ),
)

# Create logs directory if it doesn't exist
LOGS_DIR = Path.cwd() / 'logs'
if not LOGS_DIR.exists():  # Use Path.exists()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Use Path.mkdir()

# Add file handler with user-friendly formatting
LOG_FILE = LOGS_DIR / 'clean_generator_04_01.log'  # Use Path / operator
logger.add(
    LOG_FILE,
    rotation='1 day',
    compression='zip',
    format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',
)

# Tracked status message that preserves state across async tasks
_global_status = 'Processing...'
_status_lock = asyncio.Lock()


# Enhanced error handling for async functions
async def _handle_worker_error(coro, worker_id):
    """Wrapper to handle errors in async worker without losing context."""
    try:
        return await coro
    except Exception as e:
        # Get traceback with previous line info
        tb = traceback.extract_tb(e.__traceback__)
        # curr_frame = tb[-1] if tb else None # Unused variable
        prev_frame = tb[-2] if len(tb) >= 2 else None

        # Format error message with previous line context
        error_context = f'Error in worker {worker_id}: {type(e).__name__}: {e!s}'
        if prev_frame:
            error_context += f'\nPrevious line executed: {prev_frame.filename}:{prev_frame.lineno} in {prev_frame.name}: {prev_frame.line}'

        # Log the error with context
        logger.exception(error_context)

        # Re-raise to propagate
        raise


def weighted_pii_selection(num_types: int) -> list[str]:
    """Select PII types with probability proportional to their real-world frequency."""
    return random.choices(
        population=list(PII_TYPE_WEIGHTS.keys()), weights=list(PII_TYPE_WEIGHTS.values()), k=num_types
    )


# Prompt template for generating PII text
PROMPT_TEMPLATE = """
# Persona: Mestre Especialista em PII: adapta o contexto, sem alterar o valor dos PII, gerando um texto com PIIs realistas completo e coerente com o contexto e com a realidade para geração de um dataset de limpeza de dados.

Você é um especialista em gerar PIIs realistas dentro de contextos específicos para teste de modelos de limpeza de dados.
Para isso, sua **ÚNICA** tarefa é gerar um objeto JSON contendo o texto com PII conforme especificado abaixo.

## Possíveis Circunstancias da Geração (se algum valor estiver em branco, use o(s) valor(es) que não estão em branco)
**Contexto:** {context}
**Meio:** {medium}
**Intenção:** {intent}
**Estilo:** {style}

## REGRA CRÍTICA: USE APENAS OS VALORES DOS PIIs FORNECIDOS
**ATENÇÃO MÁXIMA**: Você DEVE usar SOMENTE os PIIs que forem explicitamente fornecidos na seção "Dados a Incluir".
- **PROIBIDO** adicionar qualquer PII que não esteja listado
- **PROIBIDO** inventar novos valores ou tipos de PII
- **PROIBIDO** usar valores dos de PIIs contidos nos exemplos
- **OBRIGATÓRIO** usar TODOS os PIIs fornecidos
- **OBRIGATÓRIO** usar EXATAMENTE os valores fornecidos, com absolutamente mínima alteração para fluência das frases

## Dados a Incluir
O texto deve NATURALMENTE incluir as seguintes informações pessoais (PII) (ignore os títulos dos valores, apenas use os valores):<dados_a_incluir_apenas_os_valores>
{pii_values}</dados_a_incluir_apenas_os_valores>

## PIIs Possíveis (APENAS para referência - NÃO invente valores!)
Os PIIs fornecidos acima podem ser de um ou mais dos seguintes tipos:
- e-mail
- nome/nome do meio/sobrenomes
- número de telefone (ex: (13) 3931-5015)
- número do CPF (ex: 804.797.462-90)
- número de RG (ex: 98.157.717-0)
- número do PIS (ex: 149.6590.673-6)
- número da casa ou do edifício (ex: 818, s/n)
- número de cartão de crédito (ex: 5329 1940 5320 3534)
- data de nascimento (ex: 02/11/1985)
- endereço: nome da rua, número, bairro, cidade, estado (não abreviado), UF, CEP
- CNPJ (ex: 24.795.684/0004-76)
- CEI (ex: 27.236.79409/86)
- raça/etnia
- convicção religiosa
- opinião política
- filiação a organização
- dados médicos
- dados sexuais

## Exemplos
### Siga o exemplo de geração correta!
<exemplo_correto>
input:
name: João
LAST_NAME: Silva
phone: (11) 98765-4321

output:
{{"texto_gerado_com_pii": "Olá, meu nome é João Silva. Podem me contatar pelo telefone (11) 98765-4321 para mais informações sobre o projeto."}}
</exemplo_correto>

### NÃO siga o exemplo de geração incorreta!
<exemplo_incorreto_1>
input:
name: João
LAST_NAME: Silva

output (ERRADO - adicionou PII não fornecido):
{{"texto_gerado_com_pii": "Olá, meu nome é João Silva, meu CPF é 123.456.789-00"}}
</exemplo_incorreto_1>

<exemplo_incorreto_2>
input:
name: João
LAST_NAME: Silva

output (ERRADO - incluiu labels):
{{"texto_gerado_com_pii": "Olá, meu nome é name: João e LAST_NAME: Silva"}}
</exemplo_incorreto_2>

## Regras Principais
1. **USE APENAS OS PIIs FORNECIDOS**: Não adicione NENHUM PII além dos que foram explicitamente listados em <dados_a_incluir_apenas_os_valores>"
2. **NUNCA MODIFIQUE OS VALORES DOS PIIs ALÉM DO MÍNIMO NECESSÁRIO PARA FLUÊNCIA LINGUAL**: Utilize EXATAMENTE os valores de PII fornecidos
3. **INCLUA TODOS OS PIIs**: Cada PII fornecido DEVE aparecer no texto
4. **NÃO ADICIONE NOVOS PIIs**: Se não foi fornecido um CPF, NÃO invente um. Se não foi fornecido um telefone, NÃO adicione um
5. **CONTEXTUALIZE NATURALMENTE**: Integre os PIIs de forma orgânica no texto

## Regras de Interpretação
Quando receber valores numéricos, interprete-os conforme o formato:
- Telefone: formato (XX) XXXX-XXXX ou (XX) XXXXX-XXXX
- CPF: formato XXX.XXX.XXX-XX
- RG: formato XX.XXX.XXX-X
- PIS: formato XXX.XXXXX.XX-X
- CEP: formato XXXXX-XXX
- CNPJ: formato XX.XXX.XXX/XXXX-XX
- CEI: formato XX.XXX.XXXXX/XX
- Cartão de crédito: formato XXXX XXXX XXXX XXXX
- Número de edifício: número simples ou "s/n"

## Regras de Formato
- O texto deve ter entre 2 a 8 frases (OBRIGATÓRIO)
- **NÃO adicione NENHUM tipo de PII além dos especificados**
- Mantenha um tom consistente com o contexto e o estilo
- Use as informações de forma natural e contextualizada
- Seja criativo no contexto, mas NUNCA nos valores de PII
- Retorne APENAS o objeto JSON, sem introduções ou comentários

## Avisos Críticos
- **PROIBIDO**: Inventar PIIs não fornecidos (como adicionar um e-mail quando só foi dado nome e telefone)
- **PROIBIDO**: Usar PIIs genéricos ou de exemplo que não foram fornecidos
- **OBRIGATÓRIO**: Usar TODOS e SOMENTE os PIIs listados em "<dados_a_incluir_apenas_os_valores>"
- **LEMBRE-SE**: O objetivo é testar a remoção de PIIs específicos, não criar novos

## Tarefa Final
TAREFA: Com base nessas regras, nas circumstâncias e APENAS nos PIIs fornecidos, gere um objeto JSON com a chave "texto_gerado_com_pii" contendo o texto completo **AGORA**:
"""

# Simplified prompt template for generating PII text with sample data
SIMPLIFIED_PROMPT_TEMPLATE = PROMPT_TEMPLATE

# Template for the basic PII generation prompt
PROMPT_TEMPLATE_BASIC = SIMPLIFIED_PROMPT_TEMPLATE


# Simple class for PII markers with position information
class PIIMarker:
    """Represents a marker for Personally Identifiable Information (PII) with position information.

    Attributes:
        pii_type: Type of the PII (e.g., "PERSON_NAME", "EMAIL", etc.)
        start: Starting position in the text
        end: Ending position in the text
        text: The actual PII text (optional)
    """

    def __init__(self, pii_type: str, start: int, end: int, text: str | None = None) -> None:
        self.pii_type = pii_type
        self.start = start
        self.end = end
        self.text = text

    def to_dict(self) -> dict:
        """Convert the PIIMarker to a dictionary for serialization."""
        result = {'pii_type': self.pii_type, 'start': self.start, 'end': self.end}
        if self.text:
            result['text'] = self.text
        return result


# Simple data class for sample data
class SampleData:
    """Represents sample data with text and PII markers.

    Attributes:
        text: The text containing PII
        pii_markers: list of PIIMarker objects
        metadata: Additional metadata about the sample
    """

    def __init__(self, text: str, pii_markers: list | None = None, metadata: dict | None = None) -> None:
        self.text = text
        self.pii_markers = [
            PIIMarker(
                pii_type=marker['pii_type'],
                start=marker['start'],
                end=marker['end'],
                text=text[marker['start'] : marker['end']],
            )
            for marker in (pii_markers or [])
        ]
        self.metadata = metadata or {}


class TextoComPII(BaseModel):
    """Text containing PII information with its types identified."""

    texto: str = Field(
        ..., description='Texto contendo informações pessoais sensíveis (PII) de forma natural e contextualizada'
    )
    tipos_pii: list[PIIType] = Field(..., description='lista de tipos de PII encontrados no texto')


class GeradorDeTextoComPII(BaseModel):
    """Generator output with original and masked versions.

    Example of unmasked vs. masked:

    - Unmasked: "Meu nome é João da Silva, muito prazer!"
    - Masked:   "Meu nome é [name] [LAST_NAME], muito prazer!"
    """

    texto_com_pii: TextoComPII


# Pollution ty
# ===== Utility Functions =====
def pick_one(items: list[str]) -> str:
    return random.choice(items)


def select_random_pii_items(sample_line: dict[str, any]) -> dict[str, any]:
    """Randomly select between min_items and max_items PII items from a sample line.
    biasing selection towards items relevant to the provided context and intent.
    Only selects from the allowed list of PII keys according to our rules.

    Args:
        sample_line: A dictionary containing PII items from samples.jsonl
        selected_context_key: The key of the selected context (e.g., 'consulta_medica').
        selected_intent_key: The key of the selected intent (e.g., 'diagnostico').
        min_items: Minimum number of PII items to select (default: 1)
        max_items: Maximum number of PII items to select (default: 12)

    Returns:
        A dictionary with the selected PII items
    """
    # Define the allowed keys according to our rules
    # Note: 'cei' and 'cnpj' are excluded as they are explicitly removed later in the code
    allowed_keys = [
        'CEP',
        'City_Name',
        'City_UF',  # Removed duplicate
        'State',
        'Phone',
        'Street',
        'BuildingNumber',
        'Neighborhood',
        'Name',
        'Middle_Name',
        'Surnames',
        'CPF',
        'RG',
        'PIS',
        'CNPJ',
        'CEI',
        'Email',
        'Birthdate',
        'Creditcard',  # Fixed casing to match data
        'Race_or_Ethnicity',
        'Religious_Conviction',
        'Political_Opinion',
        'Organization_Affiliation',
        'Medical_Data',
        'Sexual_Data',
    ]

    # Filter the sample line to only include allowed keys AND non-empty values
    if isinstance(sample_line, list) and sample_line:
        sample_line = sample_line[0]  # Get the first item from the list
    filtered_sample = {k: v for k, v in sample_line.items() if k.upper() in [ak.upper() for ak in allowed_keys] and v}
    # Get all available PII keys from the filtered sample
    available_keys = list(filtered_sample.keys())

    if not available_keys:
        logger.warning('No allowed keys found in sample line')
        return {}

    # Determine how many items to select (between min_items and max_items)
    num_items = min(random.randint(1, 18), len(available_keys))
    selected_pii_keys = weighted_pii_selection(num_items)
    logger.info(f'Selected {len(selected_pii_keys)} PII items from {len(available_keys)} available keys.')
    # --- Start Bias Implementation ---

    # Create a dictionary with the selected items
    selected_items = {key: filtered_sample[key] for key in selected_pii_keys}

    # Log the selected items
    logger.info(f'Selected {len(selected_items)} PII items: {", ".join(selected_pii_keys)}')

    return selected_items


# <<< START OF REPLACEMENT >>>
async def generate_random_prompt(
    selected_sample,  # Keep for potential future use or consistency, though not strictly needed now
    selected_pii_items,
) -> str:  # Now only returns the prompt string
    """Formats the prompt string using pre-selected scenario elements and PII.

    Args:
        selected_sample: Optional sample data to use (singular)
        selected_pii_items: Optional dictionary of PII items to include
        context_desc: Pre-selected context description.
        medium_desc: Pre-selected medium description.
        intent_desc: Pre-selected intent description.
        style_desc: Pre-selected style description.

    Returns:
        str: Formatted prompt string for LLM.
    """
    # Determine tone style: 50% formal, 25% informal, 25% mixed
    logger.info(f'Inside generate_random_prompt: Using selected_sample: {selected_sample}')
    logger.info(f'Inside generate_random_prompt: Using selected_pii_items: {selected_pii_items}')
    tone_choice = random.random()

    # Initialize variables to avoid potential UnboundLocalError
    context, medium, intent, style = None, None, None, None

    if 0.0 <= tone_choice < 0.25:  # 25% chance for formal
        context_key = pick_one(list(CONTEXTS['formal'].keys()))
        medium_key = pick_one(list(MEDIUMS['formal'].keys()))
        intent_key = pick_one(list(INTENTS['formal'].keys()))
        style_key = pick_one(list(STYLES['formal'].keys()))
        context = CONTEXTS['formal'].get(context_key)  # Use .get for safety
        medium = MEDIUMS['formal'].get(medium_key)
        intent = INTENTS['formal'].get(intent_key)
        style = STYLES['formal'].get(style_key)
    elif 0.25 <= tone_choice < 0.50:  # 25% chance for informal
        context_key = pick_one(list(CONTEXTS['informal'].keys()))
        medium_key = pick_one(list(MEDIUMS['informal'].keys()))
        intent_key = pick_one(list(INTENTS['informal'].keys()))
        style_key = pick_one(list(STYLES['informal'].keys()))
        context = CONTEXTS['informal'].get(context_key)
        medium = MEDIUMS['informal'].get(medium_key)
        intent = INTENTS['informal'].get(intent_key)
        style = STYLES['informal'].get(style_key)
    elif 0.50 <= tone_choice < 0.70:  # 20% chance for minimal
        minimal_category = random.choice(['context', 'medium', 'intent', 'style'])
        tone_pool = random.choice(['formal', 'informal'])  # Choose formal or informal pool
        context, medium, intent, style = '.', '.', '.', '.'  # Default to "."

        if minimal_category == 'context':
            pool_keys = list(CONTEXTS[tone_pool].keys())
            chosen_key = pick_one(pool_keys) if pool_keys else None
            context = CONTEXTS[tone_pool].get(chosen_key)
        elif minimal_category == 'medium':
            pool_keys = list(MEDIUMS[tone_pool].keys())
            chosen_key = pick_one(pool_keys) if pool_keys else None
            medium = MEDIUMS[tone_pool].get(chosen_key)
        elif minimal_category == 'intent':
            pool_keys = list(INTENTS[tone_pool].keys())
            chosen_key = pick_one(pool_keys) if pool_keys else None
            intent = INTENTS[tone_pool].get(chosen_key)
        else:  # style
            pool_keys = list(STYLES[tone_pool].keys())
            chosen_key = pick_one(pool_keys) if pool_keys else None
            style = STYLES[tone_pool].get(chosen_key)

    else:  # 0.70 <= tone_choice <= 1.00 (30% chance for mixed)
        # Mix formal and informal elements randomly
        context_tone = random.choice(['formal', 'informal'])
        medium_tone = random.choice(['formal', 'informal'])
        intent_tone = random.choice(['formal', 'informal'])
        style_tone = random.choice(['formal', 'informal'])

        context_key = pick_one(list(CONTEXTS[context_tone].keys()))
        medium_key = pick_one(list(MEDIUMS[medium_tone].keys()))
        intent_key = pick_one(list(INTENTS[intent_tone].keys()))
        style_key = pick_one(list(STYLES[style_tone].keys()))

        context = CONTEXTS[context_tone].get(context_key)
        medium = MEDIUMS[medium_tone].get(medium_key)
        intent = INTENTS[intent_tone].get(intent_key)
        style = STYLES[style_tone].get(style_key)

    if selected_pii_items:
        pii_types = list(selected_pii_items.keys())
        logger.info(f'Using provided PII types: {pii_types}')
        logger.info('DEBUG: Using selected_pii_items path - PII relationship rules may not be fully applied')

        # Create a set for easier lookup and modification
        pii_types_set = set(pii_types)
        logger.info(f'DEBUG: Initial pii_types_set: {pii_types_set}')

        # 2. When STATE or city_uf is selected, include city_name
        if ('STATE' in pii_types_set or 'city_uf' in pii_types_set) and 'city_name' not in pii_types_set:
            logger.info('DEBUG: Rule 2 check - STATE/city_uf is present without city_name')
            pii_types_set.add('city_name')
            logger.info('DEBUG: Rule 2 applied - Added city_name because STATE/city_uf is present')
            # Find a city_name value if possible
            if selected_sample and 'city_name' in selected_sample:
                value_to_add = selected_sample.get('city_name')
                if value_to_add is not None:
                    selected_pii_items['city_name'] = value_to_add
                logger.info(f'DEBUG: Added city_name value: {selected_sample["city_name"]}')

        # 3. When middle_name is selected, include name
        if 'middle_name' in pii_types_set and 'name' not in pii_types_set:
            logger.warning('DEBUG: Rule 3 check - Relationship violation detected: middle_name present without name')
            pii_types_set.add('name')
            logger.info('DEBUG: Rule 3 applied - Added name because middle_name is present')
            # Fix the relationship violation by finding a name in the samples
            if selected_sample and 'name' in selected_sample:
                value_to_add = selected_sample.get('name')
                if value_to_add is not None:
                    selected_pii_items['name'] = value_to_add
                logger.info(f'DEBUG: Added name value: {selected_sample["name"]}')
        try:
            # 5. When building_number is selected, include street
            if 'building_number' in pii_types_set and 'street' not in pii_types_set:
                logger.info('DEBUG: Rule 5 check - building_number is present without street')
                pii_types_set.add('street')
                logger.info('DEBUG: Rule 5 applied - Added street because building_number is present')
                # Find a street value if possible
                if selected_sample and 'street' in selected_sample:
                    value_to_add = selected_sample.get('street')
                    if value_to_add is not None:
                        selected_pii_items['street'] = value_to_add
                    logger.info(f'DEBUG: Added street value: {selected_sample["street"]}')

            # 6. When street is selected, include building_number
            if 'street' in pii_types_set and 'building_number' not in pii_types_set:
                logger.info('DEBUG: Rule 6 check - street is present without building_number')
                pii_types_set.add('building_number')
                logger.info('DEBUG: Rule 6 applied - Added building_number because street is present')
                # Find a building_number value if possible
                if selected_sample and 'building_number' in selected_sample:
                    value_to_add = selected_sample.get('building_number')
                    if value_to_add is not None:
                        selected_pii_items['building_number'] = value_to_add
                    logger.info(f'DEBUG: Added building_number value: {selected_sample["building_number"]}')
        except Exception as e:
            logger.exception(f'Error in Rule 5 and 6: {e!s}')

        # 8. When building_number is selected, 15% chance to change to "s/n" and 15% to "sem número"
        if 'building_number' in pii_types_set and selected_pii_items:
            building_number_chance = random.random()
            if building_number_chance < 0.15:
                # Replace the actual building number with "s/n"
                selected_pii_items['building_number'] = 's/n'
            elif building_number_chance < 0.30:
                # Replace the actual building number with "sem número"
                selected_pii_items['building_number'] = 'sem número'

        # Convert back to list
        pii_types = list(pii_types_set)
        logger.info(f'DEBUG: Final pii_types after applying rules: {pii_types}')
    else:
        # Scenario selection is now done in the worker function.
        # Initialize pii_types if not provided (shouldn't happen with current worker logic)
        pii_types = []

    # Get Portuguese descriptions -- No change needed here, handled in worker now
    pii_strings = [PII_TYPES[pii] for pii in pii_types]

    # Final validation of PII relationships -- moved to worker
    logger.info('DEBUG: Performing final validation of PII relationships (Moved to worker)')
    # validation_issues = [] # Removed, handled in worker

    # If we have selected PII items, use the simplified template
    if selected_pii_items:
        # Format PII values for the prompt - only include values, not keys
        # pii_values_str = '\n'.join([f'{v}' for k, v in selected_pii_items.items() if v])
        pii_values_str = '\n'.join([
            f"Inclua essa e as demais PII de acordo com as instruções: '{v}'"
            for k, v in selected_pii_items.items()
            if v
        ])

        # Use the passed-in descriptions
        logger.info(f'DEBUG: Using context: {context}, medium: {medium}, intent: {intent}, style: {style}')
        logger.info(f'DEBUG: Using pii_values: {pii_values_str}')
        prompt = SIMPLIFIED_PROMPT_TEMPLATE.format(
            context=context, medium=medium, intent=intent, style=style, pii_values=pii_values_str
        )
    else:
        # Use the standard template
        # Use the passed-in descriptions
        # Note: pii_types will be empty here based on current logic
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            medium=medium,
            intent=intent,
            style=style,
            pii_types=', '.join(pii_strings),  # This will be empty
        )

    # Only return the formatted prompt string
    return prompt


# <<< END OF REPLACEMENT >>>


async def load_samples_from_jsonl(file_path: str) -> list[dict[str, any]]:
    """Load sample data from a JSONL file with sophisticated error handling and path resolution.

    This function implements:
    1. Robust path resolution with fallback mechanisms
    2. Asynchronous file I/O for optimal performance
    3. Comprehensive error handling with detailed diagnostics
    4. Structured logging with contextual metadata

    Args:
        file_path: Path to the JSONL file containing sample data

    Returns:
        A list of dictionaries, each containing PII fields from the JSONL file
    """
    samples = []

    # Convert file_path to Path object early
    file_path_obj = Path(file_path)

    try:
        # If the path is not absolute, try multiple locations
        if not file_path_obj.is_absolute():  # Use Path.is_absolute()
            possible_paths_str = [
                str(file_path_obj),  # Original relative path
                str(SCRIPT_DIR / file_path_obj),  # Relative to script dir
                str(SCRIPT_DIR / 'input' / file_path_obj.name),  # In input subdir relative to script dir
            ]
            possible_paths = [Path(p) for p in possible_paths_str]

            # Find the first path that exists
            resolved_path_obj = next((p for p in possible_paths if p.exists()), None)  # Use Path.exists()

            if resolved_path_obj:
                logger.info(f'Found sample file at: {resolved_path_obj}')
                file_path_obj = resolved_path_obj  # Update file_path_obj to the resolved Path
            else:
                # Log all attempted paths for diagnostic transparency
                logger.error(f'Sample file not found. Tried paths: {", ".join(possible_paths_str)}')
                return []
        # else: file_path_obj is already absolute

        # Ensure the input directory exists with preemptive directory creation
        input_dir = file_path_obj.parent  # Use Path.parent
        if input_dir and not input_dir.exists():  # Use Path.exists()
            input_dir.mkdir(parents=True, exist_ok=True)  # Use Path.mkdir()
            logger.info(f'Created input directory: {input_dir}')
            # Removed warning here as directory creation doesn't mean file isn't found yet
            # logger.warning(f'Sample file not found: {file_path_obj}')
            # return [] # Don't return here, let the next check handle it

        # Verify file existence before attempting to open
        if not file_path_obj.exists():  # Use Path.exists()
            logger.error(f'Sample file not found: {file_path_obj}')
            return []

        async with aiofiles.open(file_path_obj, encoding='utf-8') as f:
            line_count = 0
            async for line in f:
                line_count += 1
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError:
                    # Structured error reporting with line context
                    logger.exception(f'Invalid JSON at line {line_count} in {file_path_obj}: {line[:50]}...')
                    continue
                except Exception as e:
                    logger.exception(f'Error processing line {line_count} in {file_path_obj}: {e!s}')
                    continue

        # Log success with performance metrics
        processing_rate = f'{len(samples) / line_count * 100:.1f}%' if line_count > 0 else '0%'
        logger.info(
            f'Successfully loaded {len(samples)} samples from {file_path_obj} (processed {processing_rate} of lines)'
        )
        return samples

    except FileNotFoundError:
        logger.exception(f'Sample file not found: {file_path_obj}')
        return []
    except Exception as e:
        logger.exception(f'Error loading samples from {file_path_obj}: {e!s}')
        return []


async def process_model_generation_results(generated_text, selected_samples):
    """Process model-generated text against selected samples using fuzzy matching.
    Order-agnostic validation ensuring only mapping correctness is evaluated.

    Args:
        generated_text: str - Text generated by model
        selected_samples: Union[list, str, dict] - Sample data for validation

    Returns:
        dict: Validation results dictionary with order-agnostic field mapping
    """
    mapped_fields = {}
    # Convert string representation to dictionary if necessary
    if isinstance(selected_samples, str):
        try:
            # Primary parsing with json.loads for RFC 8259 compliance
            selected_samples = json.loads(selected_samples)
        except json.JSONDecodeError:
            try:
                # Secondary parsing with ast.literal_eval for Python literals
                selected_samples = ast.literal_eval(selected_samples)
            except (SyntaxError, ValueError):
                # Fallback case for unparseable strings
                selected_samples = {}

    # Extract directly from dict or process each dict in list
    if isinstance(selected_samples, dict):
        samples_to_process = [selected_samples]
    elif isinstance(selected_samples, list):
        samples_to_process = selected_samples
    else:
        samples_to_process = []

    # Collect all field values to check against generated text
    all_fields = {}
    for sample_item in samples_to_process:
        for field_name, field_value in sample_item.items():
            if isinstance(field_value, str) and field_value.strip():
                all_fields[field_name] = field_value

    # Process each field against the generated text
    for field_name, field_value in all_fields.items():
        # O(n) Boyer-Moore string matching
        if field_value in generated_text:
            mapped_fields[field_name] = field_value
            continue

        # Split into words and get window size
        field_words = field_value.split()
        window_size = len(field_words)

        # Create sliding windows from generated text
        generated_words = generated_text.split()
        max_score = 0

        for i in range(len(generated_words) - window_size + 1):
            window = ' '.join(generated_words[i : i + window_size])
            # Use appropriate fuzzy algorithm
            algorithm = fuzz.token_set_ratio if window_size > 1 else fuzz.partial_ratio
            score = algorithm(window, field_value)
            max_score = max(max_score, score)

        if max_score >= 90:  # Empirically determined threshold
            mapped_fields[field_name] = field_value

    # Calculate missing fields post-processing
    missing_fields = [field_name for field_name in all_fields if field_name not in mapped_fields]

    # Identify pollution candidates per specification
    pollution_candidates = [
        field_name for field_name, field_value in mapped_fields.items() if generated_text.count(field_value) > 1
    ]

    # Calculate mapping success - order independent
    successfully_mapped = len(missing_fields) == 0

    if not successfully_mapped:
        logger.info(f'generated_text: {generated_text}')
        logger.info(f'selected_samples: {selected_samples}')
        logger.info(f'Mapped fields: {mapped_fields}')
        logger.info(f'Missing fields: {missing_fields}')
        logger.info(f'Pollution candidates: {pollution_candidates}')

    return {
        'successfully_mapped': successfully_mapped,
        'mapped_fields': mapped_fields,
        'missing_fields': missing_fields,
        'pollution_candidates': pollution_candidates,
    }


def generate_random_email(sample_data: dict) -> str:
    """Generate a random email using name or surnames from sample data.

    Args:
        sample_data: Dictionary containing sample data with name and surnames fields

    Returns:
        A randomly generated email address
    """
    # List of email providers
    providers = [
        'gmail.com',
        'hotmail.com',
        'yahoo.com',
        'outlook.com',
        'icloud.com',
        'protonmail.com',
        'aol.com',
        'msn.com',
        'live.com',
        'mail.com',
        'yahoo.com.br',
        'hotmail.com.br',
        'outlook.com.br',
        # Brazilian providers
        'uol.com.br',
        'bol.com.br',
        'terra.com.br',
        'globo.com',
        'ig.com.br',
        # Educational providers
        'usp.edu.br',
        'unicamp.edu.br',
        'unifesp.edu.br',
        'puc-rio.edu.br',
        'ufmg.edu.br',
        # Additional providers
        'zoho.com',
        'gmx.com',
        'yandex.com',
        'tutanota.com',
        'fastmail.com',
        'hey.com',
        'mailchimp.com',
        'hubspot.com',
        'pm.me',
        'duck.com',
        # Additional Brazilian providers
        'r7.com',
        'zipmail.com.br',
        'folha.com.br',
        'estadao.com.br',
        'brturbo.com.br',
        # Additional educational providers
        'ufrj.edu.br',
        'unb.edu.br',
        'ufpr.edu.br',
        'ufc.edu.br',
        'ufba.edu.br',
    ]

    # Get name components
    name = sample_data.get('name', '')
    middle_name = sample_data.get('middle_name', '')
    surnames = sample_data.get('surnames', '')

    # Clean and normalize the name components
    name = name.lower().strip().replace(' ', '')
    middle_name = middle_name.lower().strip().replace(' ', '')
    surnames_parts = surnames.lower().split()
    surnames_cleaned = [part.replace(' ', '') for part in surnames_parts]

    # Generate username components
    username_options = [
        name,
        f'{name}{random.choice(["", ".", "_"])}{middle_name[0] if middle_name else ""}',
        f'{name}{random.randint(1, 999)}',
        f'{name}{random.choice(["", ".", "_"])}{surnames_cleaned[0] if surnames_cleaned else ""}',
        f'{name[0]}{random.choice(["", ".", "_"])}{surnames_cleaned[0] if surnames_cleaned else ""}',
    ]

    # Add more options if surnames are available
    if surnames_cleaned:
        username_options.extend([
            surnames_cleaned[0],
            f'{surnames_cleaned[0]}{random.randint(1, 999)}',
            f'{name[0]}{random.choice(["", ".", "_"])}{surnames_cleaned[0]}',
            f'{surnames_cleaned[0]}{random.choice(["", ".", "_"])}{surnames_cleaned[-1] if len(surnames_cleaned) > 1 else ""}',
        ])

    # Select a random username
    username = random.choice(username_options)

    # Select a random provider
    if random.random() < 0.5:
        # 50% chance for .br or .edu.br domain
        provider = random.choice([p for p in providers if p.endswith('.br')])
    else:
        # 50% chance for .com domain
        provider = random.choice([p for p in providers if p.endswith('.com')])

    # Generate the email
    return f'{username}@{provider}'


def get_random_weighted_samples(sample_corpus: list, min_count: int = 2, max_count: int = 12) -> list:
    """Selects a SINGLE random sample_line from the sample_corpus.  This now returns a list
    containing only the selected sample, to maintain compatibility with the
    existing code structure that expects a list.

    Args:
        sample_corpus: The corpus of sample lines.
        min_count:  Not used. Kept for compatibility.
        max_count:  Not used. Kept for compatibility.

    Returns:
        A list containing the single, randomly selected sample_line.
    """
    if not sample_corpus:
        logger.error('Sampling operation failed: Empty corpus provided')
        return []

    # Phase 1: Select a single sample line
    selected_sample_line = random.choice(sample_corpus)
    logger.info(f'Selected sample line: {selected_sample_line}')

    return [selected_sample_line]  # Return as a list


async def worker(
    queue: asyncio.Queue,
    results: list,
    results_saver: ResultsSaver,
    pollute_data: bool,
    pollution_rate: float,
    pollution_types: list,
    samples_source: list,
    session: aiohttp.ClientSession,
    easy: int | None = None,
) -> None:
    """Asynchronous worker that processes items from a queue to generate synthetic PII data.

    This worker fetches tasks from a queue, selects sample data, generates PII content,
    and handles the results. It implements error handling, metrics tracking, and supports
    optional data pollution for testing purposes.

    Args:
        queue: Asynchronous queue containing indices of tasks to process
        results: Shared list where results are stored at their corresponding indices
        results_saver: Object responsible for saving results to persistent storage
        pollute_data: Flag indicating whether to apply data pollution
        pollution_rate: Percentage of data to pollute (0.0-1.0)
        pollution_types: List of pollution types to apply
        samples_source: List of sample data to use for generation
        easy: Optional difficulty level parameter
    """
    # Initialize metrics tracking
    global _rejected_sample_counter, _generated_sample_counter  # CHANGED: Added generated counter
    metrics = {
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'retried': 0,
        'pollution_applied': 0,
        'avg_processing_time_ms': 0,
        'pii_mapping_ratios': [],
        'sample_counts': [],
    }

    # Set default model parameters
    model_id = DEFAULT_MODEL_ID
    temperature = 1

    # Initialize batch for result persistence
    batch = []
    batch_size = 10  # Default batch size for persistence

    # Process items from the queue
    while True:
        item = None  # Initialize item to track successful get()
        try:
            # Get the next index from the queue
            item = await queue.get()
            index = item  # Process item only if get() succeeded
            metrics['processed'] += 1
            task_start_time = datetime.datetime.now(datetime.UTC)

            # Handle potential retry wrapper
            original_index = index
            if hasattr(index, 'index'):
                index = index.index

            try:
                # Verify sample data is available
                if not samples_source:
                    async with _rejected_sample_lock:
                        _rejected_sample_counter += 1
                        rejection_count = _rejected_sample_counter
                    # NEW: Display total stats
                    async with _generated_sample_lock:
                        generated_count = _generated_sample_counter
                    logger.error(
                        f'Worker halted: No sample data available | Total rejected: {rejection_count} | Total generated: {generated_count}'
                    )
                    logger.error(
                        f'REJECTED SAMPLE #{rejection_count} - Index: {index}, Reason: No sample data available'
                    )
                    results[index] = None
                    metrics['failed'] += 1
                    continue

                # STEP 1: Select a single sample from the corpus
                selected_sample = get_random_weighted_samples(samples_source)
                logger.info(f'Selected sample line: {selected_sample}')  # Returns a list with one item
                if isinstance(selected_sample, list) and selected_sample:
                    sample_dict = selected_sample[0]
                    # Add EMAIL if it doesn't exist
                    if 'email' not in sample_dict:
                        email = generate_random_email(sample_dict)
                        sample_dict['email'] = email
                        logger.info(f'Added generated email to sample: {email}')
                    if 'neighborhood' in sample_dict and not sample_dict['neighborhood']:
                        sample_dict['neighborhood'] = 'centro' if random.random() < 0.3 else 'Centro'

                # Now proceed with PII selection
                selected_pii_items = select_random_pii_items(selected_sample)

                # Log detailed information about selected samples and PII items for debugging
                # logger.info(f'Selected sample: {json.dumps(selected_sample, cls=PydanticJSONEncoder, ensure_ascii=False)}'[:200]) # Reduce verbosity
                logger.info(
                    f'Selected PII items: {json.dumps(selected_pii_items, cls=PydanticJSONEncoder, ensure_ascii=False)}'[
                        :200
                    ]
                )

                # Guard against degenerate selection
                if not selected_pii_items:
                    async with _rejected_sample_lock:
                        _rejected_sample_counter += 1
                        rejection_count = _rejected_sample_counter
                    # NEW: Display total stats
                    async with _generated_sample_lock:
                        generated_count = _generated_sample_counter
                    logger.error(
                        f'Execution halted: PII extraction failure | Total rejected: {rejection_count} | Total generated: {generated_count}'
                    )
                    logger.error(
                        f'REJECTED SAMPLE #{rejection_count} - Index: {index}, Reason: PII extraction failure, Sample: {json.dumps(selected_sample, cls=PydanticJSONEncoder, ensure_ascii=False)[:200]}'
                    )
                    results[index] = None
                    metrics['failed'] += 1
                    continue

                # Log selection metrics for observability
                logger.info(f'Selected 1 sample yielding {len(selected_pii_items)} PII entities'[:100])
                metrics['sample_counts'].append(1)  # Always 1 now

                # Log PII items before generating prompt
                logger.info(f'DEBUG: Selected PII items before prompt generation: {list(selected_pii_items.keys())}')

                # STEP 3: Generate optimized prompt with domain-specific contextualization
                # Now pass the pre-selected scenario descriptions to generate_random_prompt
                # We expect generate_random_prompt to just format the string now
                prompt = await generate_random_prompt(
                    selected_sample=selected_sample, selected_pii_items=selected_pii_items
                )

                # Log PII items after generating prompt to see if they changed
                logger.info(f'DEBUG: Selected PII items after prompt generation: {list(selected_pii_items.keys())}')

                # Log the generated prompt for debugging and traceability
                # logger.info(f'Generated prompt: {prompt}'[:100]) # Reduce log verbosity

                # --- Add Log: Before Generation ---
                current_time_str = datetime.datetime.now(datetime.UTC).strftime('%a:%d:%Y %H:%M:%S')
                logger.info(
                    f'The masks given at {current_time_str} were: {json.dumps(selected_pii_items, cls=PydanticJSONEncoder, ensure_ascii=False)}'
                )
                # --- End Log ---

                # Maximum generative attempts with progressive backoff strategy
                max_attempts = 10
                attempt = 0
                generated_text = None
                masked_text = None
                model_id = DEFAULT_MODEL_ID
                # Generation loop with quality validation
                while attempt < max_attempts and not generated_text:
                    attempt += 1
                    logger.debug(f'Generative attempt {attempt}/{max_attempts}'[:200])

                    try:
                        # STEP 3: Invoke generative model with optimized prompt
                        # This represents the core transaction with the AI system
                        generated_text, masked_text, model_id = await use_model_to_generate_PII(
                            session, prompt, selected_pii_items
                        )

                        # NEW: Increment generated counter for each attempt
                        async with _generated_sample_lock:
                            _generated_sample_counter += 1
                            generated_count = _generated_sample_counter
                        async with _rejected_sample_lock:
                            rejection_count = _rejected_sample_counter
                        logger.info(
                            f'Generation attempt {attempt} | Total rejected: {rejection_count} | Total generated: {generated_count}'
                        )

                        # Validate that both generated_text and masked_text are not None
                        try:
                            if generated_text is None or masked_text is None:
                                logger.error(
                                    f'Generation failed: generated_text or masked_text is None. generated_text: {generated_text}, masked_text: {masked_text}'
                                )
                                generated_text = None  # Reset to force another attempt
                                continue
                            logger.info(f'Generated text: {generated_text}'[:200])
                            logger.info(f'Masked text: {masked_text}'[:200])
                        except Exception as e:
                            logger.exception(f'Error validating generated text: {e}')
                            generated_text = None  # Reset to force another attempt
                            continue
                        # STEP 4: Execute comprehensive mapping between input and output
                        # This implements the critical validation of what was sent vs. what was generated
                        try:
                            mapping_result = await process_model_generation_results(generated_text, selected_pii_items)
                        except Exception as e:
                            logger.exception(f'Error processing model generation results: {e}')
                            generated_text = None  # Reset to force another attempt
                            continue

                        # STEP 5: Evaluate mapping integrity with progressive acceptance criteria
                        if mapping_result['successfully_mapped']:
                            metrics['pii_mapping_ratios'].append(1.0)
                            logger.info(f'mapping_result: {mapping_result}')
                        if mapping_result.get('missing_fields') and len(mapping_result['missing_fields']) > 0:
                            # Partial mapping scenario with quantitative evaluation
                            missing_fields = mapping_result['missing_fields']
                            async with _rejected_sample_lock:
                                _rejected_sample_counter += 1
                                rejection_count = _rejected_sample_counter
                            # NEW: Display total stats
                            async with _generated_sample_lock:
                                generated_count = _generated_sample_counter
                            logger.warning(f'Partial mapping detected: {(missing_fields)}')
                            logger.info(f'masked_text text: {masked_text}')
                            logger.error(
                                f'REJECTED SAMPLE #{rejection_count} - Index: {index}, Attempt: {attempt}, Reason: Missing fields in mapping, Missing: {missing_fields}, Generated text: {generated_text[:200]} | Total rejected: {rejection_count} | Total generated: {generated_count}'
                            )
                            generated_text = None  # Reset generated_text to force another attempt
                            continue

                    except Exception as e:
                        logger.exception(f'Generation attempt failure ({attempt}/{max_attempts}): {e}')
                        continue

                # Initialize result_pollution to avoid reference errors

                # --- Add Log: Before Masking (Final) ---
                # Log the final generated text and the PII dict used for the final masked output
                if generated_text:  # Only log if generation was successful
                    logger.info(f'Text before final masking: {generated_text}')
                    logger.info(
                        f'PII dict for final masking: {json.dumps(selected_pii_items, cls=PydanticJSONEncoder, ensure_ascii=False)}'
                    )
                if not generated_text:
                    async with _rejected_sample_lock:
                        _rejected_sample_counter += 1
                        rejection_count = _rejected_sample_counter
                    # NEW: Display total stats
                    async with _generated_sample_lock:
                        generated_count = _generated_sample_counter
                    logger.error(
                        f'REJECTED SAMPLE #{rejection_count} - Index: {index}, Reason: Failed after {max_attempts} generation attempts, PII items: {list(selected_pii_items.keys())} | Total rejected: {rejection_count} | Total generated: {generated_count}'
                    )
                # STEP 6: Compose comprehensive result entity with full provenance
                result = {
                    '_id': str(uuid.uuid4()),
                    'generated_sample_PII': generated_text,
                    'items_used': [selected_pii_items],  # Still a list for consistency
                    'generated_sample_masked': masked_text,
                    'masks': mapping_result['mapped_fields'],
                    'timestamp': datetime.datetime.now(datetime.UTC).isoformat(),
                    '_model_clean_generator': model_id,  # Add model ID here
                    'temperature': temperature,
                }

                # Store result with canonical index resolution
                if hasattr(original_index, 'index'):
                    results[original_index.index] = result
                else:
                    results[index] = result

                # Append to persistence batch with transaction semantics
                batch.append(result)
                metrics['successful'] += 1

                # Calculate processing time for performance metrics
                task_duration = (datetime.datetime.now(datetime.UTC) - task_start_time).total_seconds() * 1000  # ms
                metrics['avg_processing_time_ms'] = (
                    metrics['avg_processing_time_ms'] * (metrics['successful'] - 1) + task_duration
                ) / metrics['successful']

                # Execute batch persistence with optimal sizing
                if len(batch) >= batch_size:
                    await results_saver.save_batch(batch)
                    batch = []

                # Emit periodic operational metrics
                if metrics['processed'] % 20 == 0:
                    success_rate = metrics['successful'] / metrics['processed']
                    avg_items = (
                        sum(metrics['sample_counts']) / len(metrics['sample_counts'])
                        if metrics['sample_counts']
                        else 0
                    )  # Will be 1
                    avg_mapping = (
                        sum(metrics['pii_mapping_ratios']) / len(metrics['pii_mapping_ratios'])
                        if metrics['pii_mapping_ratios']
                        else 0
                    )

                    # NEW: Show total stats in periodic metrics
                    async with _rejected_sample_lock:
                        rejection_count = _rejected_sample_counter
                    async with _generated_sample_lock:
                        generated_count = _generated_sample_counter

                    logger.info(
                        f'Worker operational metrics: processed={metrics["processed"]}, '
                        f'success_rate={success_rate:.1%}, avg_time={metrics["avg_processing_time_ms"]:.1f}ms, '
                        f'avg_items={avg_items:.1f}, mapping_quality={avg_mapping:.1%} | '
                        f'Total rejected: {rejection_count} | Total generated: {generated_count}'
                    )

            except Exception as e:
                # Root exception handler with comprehensive logging
                logger.exception(f'Task processing failure ({index}): {e}')
                results[index] = None  # Mark as failed
                metrics['failed'] += 1

        except asyncio.CancelledError:
            # Handle worker cancellation with graceful resource cleanup
            logger.info('Worker received cancellation signal - initiating cleanup')

            # Persist any remaining items in the batch
            if batch:
                try:
                    await results_saver.save_batch(batch)
                    logger.info(f'Successfully persisted final batch of {len(batch)} items during cancellation')
                except Exception as save_exc:
                    logger.exception(f'Error saving remaining batch during cancellation: {save_exc}')
            break  # Exit loop after cleanup
        except Exception as e:
            # Catch potential errors during queue.get() or initial setup before inner try
            logger.exception(f'Error in worker main loop (outside item processing): {e}')
            # Decide whether to break or continue based on error type
            break  # Safer to break on unexpected errors
        finally:
            # Ensure task_done() is called exactly once per successful queue.get()
            if item is not None:
                queue.task_done()

    # --- Loop End ---

    # Emit final worker metrics after the loop has finished
    success_rate = metrics['successful'] / metrics['processed'] if metrics['processed'] > 0 else 0
    # NEW: Show final totals
    async with _rejected_sample_lock:
        final_rejected = _rejected_sample_counter
    async with _generated_sample_lock:
        final_generated = _generated_sample_counter
    logger.info(
        f'Worker terminating with metrics: processed={metrics["processed"]}, success_rate={success_rate:.1%}, retries={metrics["retried"]} | '
        f'FINAL TOTALS - Rejected: {final_rejected} | Generated: {final_generated}'
    )


async def process_with_workers(
    count: int = typer.Option(DEFAULT_COUNT, '--count', '-c', help='Number of PII samples to generate'),
    output_file: str = DEFAULT_OUTPUT_FILE,
    samples_source: list[dict[str, any]] | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    easy: int | None = None,
) -> list[dict[str, any]]:
    """Orchestrate the processing of multiple generation tasks using a sophisticated worker pool.

    This function implements a high-performance concurrent processing architecture:
    1. Dynamic worker pool sizing based on workload and system capabilities
    2. Asynchronous task queue with optimized producer-consumer pattern
    3. Incremental result persistence with atomic batch operations
    4. Comprehensive metrics collection and performance monitoring
    5. Graceful error handling and worker lifecycle management

    The core workflow enhancements include:
    1. Randomly determining how many samples to use (2-10) for each generation
    2. Selecting samples using weighted random selection based on data richness
    3. Comparing what was sent to the model with what was generated
    4. Mapping input keys to generated content with validation

    Args:
        count: Number of samples to generate
        output_file: Path to the output file
        should_pollute: Whether to apply data pollution
        pollution_rate: Percentage of samples to pollute
        pollution_types: Types of pollution to apply
        samples_source: list of dictionaries from samples.jsonl
        num_workers: Number of concurrent workers
        batch_size: Number of results to save in each batch
        easy: Optional shortcut mode count

    Returns:
        list of generated results
    """
    connector = aiohttp.TCPConnector(
        limit=100,  # Total pool - much lower than concurrency
        limit_per_host=30,  # Conservative per-host limit for localhost
        keepalive_timeout=10,  # Short keepalive - cycle connections frequently
        enable_cleanup_closed=True,  # Critical: immediately clean closed connections
        force_close=False,  # Allow reuse but with strict limits
        ttl_dns_cache=300,
        use_dns_cache=True,
        ssl=False,
    )

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(
            total=600,  # Long timeout for generation tasks
            connect=60,  # Generous connect timeout
            sock_read=300,  # Long read timeout for LLM generation
        ),
    ) as session:
        logger.info('Created aiohttp session with high concurrency limits.')

        # --- START OF INDENTED BLOCK ---
        # The following logic is now correctly inside the `async with` block.
        if count is None:
            count = 0
        output_dir = Path(output_file).parent  # Use Path
        if output_dir and not output_dir.exists():  # Use Path.exists()
            output_dir.mkdir(parents=True, exist_ok=True)  # Use Path.mkdir()
            logger.info(f'Created output directory: {output_dir}')

        # Initialize the asynchronous processing infrastructure
        queue = asyncio.Queue()
        results = [None] * count

        # Initialize the results saver with the specified output file
        results_saver = ResultsSaver(output_file)

        # Populate the task queue with work items
        for i in range(count):
            await queue.put(i)

        # Configure and launch the worker pool with optimal sizing
        workers = []
        # Calculate optimal worker count based on workload and system constraints
        worker_count = min(num_workers, count)
        logger.info(f'Using {worker_count} workers for {count} items')

        if worker_count < 20 and count >= 20:
            logger.warning(
                f'Using only {worker_count} workers for {count} items. '
                f'For optimal throughput with this workload, consider using at least 20 workers.'
            )

        # Performance metrics for monitoring and diagnostics
        start_time = datetime.datetime.now(datetime.UTC)
        performance_metrics = {
            'start_time': start_time,
            'worker_count': worker_count,
            'tasks_total': count,
            'tasks_completed': 0,
            'last_progress_log': start_time,
        }

        # Launch worker tasks with proper error isolation
        for worker_id in range(worker_count):
            worker_task = asyncio.create_task(
                worker(
                    queue,
                    results,
                    results_saver,
                    should_pollute,
                    pollution_rate,
                    pollution_types,
                    samples_source,
                    session,  # PASS THE CREATED SESSION
                    easy,
                )
            )
            workers.append(worker_task)
            logger.debug(f'Started worker {worker_id + 1}/{worker_count}')

        # Monitor progress with periodic status updates
        async def monitor_progress() -> None:
            while not queue.empty():
                try:
                    tasks_remaining = queue.qsize()
                    tasks_completed = count - tasks_remaining
                    performance_metrics['tasks_completed'] = tasks_completed

                    # Calculate completion percentage and ETA
                    completion_percentage = tasks_completed / count * 100 if count > 0 else 0
                    elapsed = (datetime.datetime.now(datetime.UTC) - start_time).total_seconds()

                    # Only estimate ETA if we have meaningful progress
                    if tasks_completed > 0:
                        estimated_total_seconds = elapsed / tasks_completed * count
                        remaining_seconds = max(0, estimated_total_seconds - elapsed)
                        eta = datetime.timedelta(seconds=int(remaining_seconds))
                        eta_str = f', ETA: {eta}'
                    else:
                        eta_str = ''

                    # Log progress every 30 seconds or 10% completion, whichever comes first
                    current_time = datetime.datetime.now(datetime.UTC)
                    time_since_last_log = (current_time - performance_metrics['last_progress_log']).total_seconds()

                    if time_since_last_log > 30 or (
                        tasks_completed > 0 and tasks_completed % max(1, count // 10) == 0
                    ):
                        logger.info(f'Progress: {tasks_completed}/{count} ({completion_percentage:.1f}%){eta_str}')
                        performance_metrics['last_progress_log'] = current_time

                    await asyncio.sleep(min(1, 8))  # Check progress every 5 seconds
                except asyncio.CancelledError:
                    logger.info('Progress monitor cancelled')
                    break

        # Start the progress monitor
        monitor_task = asyncio.create_task(monitor_progress())

        try:
            # Wait for all tasks to be processed with proper timeout handling
            await asyncio.wait_for(queue.join(), timeout=None)
        except (TimeoutError, KeyboardInterrupt, asyncio.CancelledError):
            if isinstance(sys.exc_info()[1], KeyboardInterrupt):
                logger.warning('Keyboard interrupt received - stopping processing')
            else:
                logger.warning('Processing interrupted - collecting partial results')
        finally:
            # Clean cancellation of monitor task
            if not monitor_task.done():
                monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor_task

            # Graceful worker termination with proper resource cleanup
            for w in workers:
                if not w.done():
                    w.cancel()

            # Wait for all workers to cleanly terminate
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)

        # Process and validate results
        valid_results = [r for r in results if r is not None]
        success_rate = len(valid_results) / count if count > 0 else 0

        # Calculate completion time and performance metrics
        end_time = datetime.datetime.now(datetime.UTC)
        duration = (end_time - start_time).total_seconds()
        items_per_second = len(valid_results) / duration if duration > 0 else 0

        # Log detailed performance summary
        logger.info(f'Processing completed in {duration:.2f}s ({items_per_second:.2f} items/sec)')
        logger.info(f'Success rate: {success_rate:.1%} ({len(valid_results)}/{count} items)')

        # Display performance metrics in a structured format
        if RICH_AVAILABLE and console:
            status_color = 'green' if success_rate >= 0.95 else 'yellow' if success_rate >= 0.8 else 'red'
            console.print(
                Panel.fit(
                    f'[bold {status_color}]Processing Summary[/]\n\n'
                    f'[bold]Requested:[/] {count} samples\n'
                    f'[bold]Successful:[/] {len(valid_results)} samples\n'
                    f'[bold]Failed:[/] {count - len(valid_results)} samples\n'
                    f'[bold]Success rate:[/] {success_rate:.1%}\n'
                    f'[bold]Duration:[/] {duration:.2f} seconds\n'
                    f'[bold]Throughput:[/] {items_per_second:.2f} items/sec\n'
                    f'[bold]Workers used:[/] {worker_count}\n'
                    f'[bold]Output file:[/] {output_file}',
                    title='Generation Results',
                    border_style=status_color,
                    padding=(1, 2),
                )
            )

        # Check for critical success rate thresholds with appropriate error propagation
        if len(valid_results) < count:
            if success_rate < 0.5:
                error_msg = f'Generation completed with critically low success rate: {success_rate:.1%}'
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            if success_rate < 0.8:
                logger.warning(f'Generation completed with suboptimal success rate: {success_rate:.1%}')

        return valid_results


class ResultsSaver:
    """Thread-safe asynchronous persistence manager for incremental result serialization.

    This class implements a sophisticated persistence strategy with:
    - Atomic batch operations using exclusive lock semantics
    - Polymorphic serialization supporting heterogeneous object graphs
    - Optimized I/O patterns with progressive buffer flushing
    - Comprehensive telemetry with fine-grained instrumentation
    - Graceful degradation with polyglot fallback mechanisms

    The implementation follows the Repository pattern with Transaction Script
    behaviors layered on a reactive persistence framework.
    """

    def __init__(self, output_file: str) -> None:
        """Initialize the persistence manager with a designated output path.

        Performs preemptive directory creation and establishes an async
        coordination primitive for exclusive writer access.

        Args:
            output_file: Fully qualified path to persistence target
        """
        self.output_file = output_file
        self.lock = asyncio.Lock()
        self.metrics = {
            'batches_saved': 0,
            'total_items_saved': 0,
            'last_save_time': None,
            'errors': 0,
            'recovery_attempts': 0,
            'recovery_successes': 0,
            'write_latencies_ms': [],
        }

        # Ensure directory hierarchy exists with proper permissioning
        output_dir = Path(output_file).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'Created output directory hierarchy: {output_dir}')

        # Initialize the output file without truncation to support append semantics
        if not Path(output_file).exists():
            Path(output_file).touch()
            logger.info(f'Initialized empty persistence target: {output_file}')

    async def save_batch(self, batch: list[any]) -> None:
        """Persist a batch of results with transactional semantics.

        This method implements a sophisticated non-blocking I/O strategy with:
        1. Exclusive resource acquisition via async mutex
        2. Polymorphic serialization based on type introspection
        3. Progressive buffer flushing with explicit durability semantics
        4. Graceful degradation to synchronous fallback on async failure
        5. Comprehensive telemetry collection for operation profiling

        Args:
            batch: Collection of serializable entities to persist

        Raises:
            IOError: If persistence fails after recovery attempts
        """
        if not batch:
            return

        operation_start = datetime.datetime.now(datetime.UTC)

        async with self.lock:
            try:
                # Use non-blocking I/O with explicit buffer management
                async with aiofiles.open(self.output_file, 'a', encoding='utf-8') as f:
                    for item in batch:
                        json_str = json.dumps(item, ensure_ascii=False, cls=PydanticJSONEncoder) + '\n'
                        await f.write(json_str)

                    # Explicit buffer flush with durability guarantee
                    await f.flush()

                # Update operational telemetry
                self.metrics['batches_saved'] += 1
                self.metrics['total_items_saved'] += len(batch)
                self.metrics['last_save_time'] = datetime.datetime.now(datetime.UTC)

                # Calculate and record operation latency
                operation_duration_ms = (self.metrics['last_save_time'] - operation_start).total_seconds() * 1000
                self.metrics['write_latencies_ms'].append(operation_duration_ms)

                # Log with detailed operational metrics
                avg_latency = (
                    sum(self.metrics['write_latencies_ms']) / len(self.metrics['write_latencies_ms'])
                    if self.metrics['write_latencies_ms']
                    else 0
                )
                logger.info(
                    f'Persisted batch of {len(batch)} entities in {operation_duration_ms:.2f}ms (avg: {avg_latency:.2f}ms)'
                )

                # Emit periodic cumulative metrics for observability
                if self.metrics['batches_saved'] % 10 == 0:
                    throughput = (
                        self.metrics['total_items_saved'] / len(self.metrics['write_latencies_ms']) * 1000
                        if self.metrics['write_latencies_ms']
                        else 0
                    )
                    logger.info(
                        f'Persistence metrics: {self.metrics["batches_saved"]} batches, '
                        f'{self.metrics["total_items_saved"]} total entities, '
                        f'{throughput:.2f} entities/sec, '
                        f'{self.metrics["errors"]} errors'
                    )

            except Exception as e:
                self.metrics['errors'] += 1
                logger.exception(f'Error persisting batch to {self.output_file}: {e}')

                # Implement graceful degradation with synchronous fallback
                self.metrics['recovery_attempts'] += 1
                try:
                    logger.warning('Initiating synchronous persistence recovery...')
                    async with aiofiles.open(self.output_file, 'a', encoding='utf-8') as f:
                        for item in batch:
                            # Replicate type-driven serialization logic
                            if hasattr(item, 'model_dump'):
                                item_dict = item.model_dump()
                            elif hasattr(item, 'dict'):
                                item_dict = item.dict()
                            else:
                                item_dict = item
                            # Write with explicit flush for durability
                            f.write(json.dumps(item_dict, ensure_ascii=False, cls=PydanticJSONEncoder) + '\n')
                        await f.flush()

                    # Record recovery success metrics
                    self.metrics['recovery_successes'] += 1
                    logger.info(f'Synchronous recovery successful: persisted batch of {len(batch)} entities')

                    # Update operational telemetry post-recovery
                    self.metrics['batches_saved'] += 1
                    self.metrics['total_items_saved'] += len(batch)
                    self.metrics['last_save_time'] = datetime.datetime.now(datetime.UTC)

                except Exception as recovery_e:
                    logger.exception(f'Persistence recovery failed: {recovery_e}')
                    msg = f'Critical persistence failure after recovery attempt: {recovery_e}'
                    raise OSError(msg) from e

    async def finalize(self) -> dict[str, any]:
        """Finalize the persistence session and return operational metrics.

        Performs resource cleanup, durability verification, and
        generates comprehensive operational telemetry.

        Returns:
            dictionary of operational metrics including throughput,
            latency distributions, and error rates.
        """
        async with self.lock:
            # Calculate advanced operational metrics
            self.metrics['final_time'] = datetime.datetime.now(datetime.UTC)  # Corrected

            if self.metrics['batches_saved'] > 0:
                # Calculate latency percentiles if we have data
                if self.metrics['write_latencies_ms']:
                    sorted_latencies = sorted(self.metrics['write_latencies_ms'])
                    self.metrics['p50_latency_ms'] = sorted_latencies[len(sorted_latencies) // 2]
                    self.metrics['p90_latency_ms'] = sorted_latencies[int(len(sorted_latencies) * 0.9)]
                    self.metrics['p99_latency_ms'] = sorted_latencies[int(len(sorted_latencies) * 0.99)]

                # Calculate operational efficiency metrics
                total_duration = (
                    self.metrics['final_time'] - self.metrics.get('start_time', self.metrics['last_save_time'])
                ).total_seconds()
                self.metrics['throughput_items_per_sec'] = (
                    self.metrics['total_items_saved'] / total_duration if total_duration > 0 else 0
                )
                self.metrics['error_rate'] = self.metrics['errors'] / self.metrics['batches_saved']
                self.metrics['recovery_rate'] = (
                    self.metrics['recovery_successes'] / self.metrics['recovery_attempts']
                    if self.metrics['recovery_attempts'] > 0
                    else 1.0
                )

            logger.info(
                f'Persistence session finalized: {self.metrics["total_items_saved"]} entities in {self.output_file}'
            )
            return self.metrics.copy()  # Return immutable copy for caller analysis


async def qwen_gerador_PII(
    session: aiohttp.ClientSession, prompt: str, pii_values_dict: dict[str, str], retry_count: int = 0
) -> tuple[str, str]:
    """Generate PII text using Qwen with xgrammar EBNF for structured output."""
    # Define EBNF grammar for JSON output with the specific structure we need
    ebnf_grammar = """
root ::= json_object
json_object ::= "{" ws "\\\"texto_gerado_com_pii\\\"" ws ":" ws json_string ws "}"
json_string ::= "\\"" string_content "\\""
string_content ::= ([^"\\\\] | "\\\\" | "\\\"" | "\\\\n" | "\\\\r" | "\\\\t")*
ws ::= [ \\t\\n\\r]*
"""

    max_retries = 10
    backoff_factor = 1.5
    # Build the prompt that includes JSON structure
    enhanced_prompt = f"""{prompt}

    Formate a sua resposta como um objeto JSON da seguinte maneira:
    {{"texto_gerado_com_pii": "INSIRA_O_TEXTO_AQUI"}}

    Substitua INSIRA_O_TEXTO_AQUI com o texto gerado estritamente conforme as instruções acima."""

    while retry_count < max_retries:
        try:
            payload = {
                'model': DEFAULT_MODEL_ID,  # CHANGED: Now uses the constant
                'messages': [{'role': 'user', 'content': enhanced_prompt}],
                'temperature': DEFAULT_TEMPERATURE,
                'top_p': DEFAULT_TOP_P,  # CHANGED: Now uses the constant
                'top_k': 20,
                'min_p': 0,
                'max_tokens': 4096,
                'stream': False,
                'extra_body': {'grammar': ebnf_grammar, 'grammar_type': 'ebnf'},
            }

            async with session.post(
                'http://localhost:30000/v1/chat/completions',
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
                ssl=False,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(f'HTTP {response.status}: {error_text} (attempt {retry_count + 1})')
                    retry_count += 1
                    continue

                result = await response.json()
                content = result['choices'][0]['message']['content'].strip()
                logger.debug(f'Raw response: {content}')

                # Parse the guaranteed valid JSON from xgrammar
                try:
                    generated_data = json.loads(content)

                    # Extract the texto_gerado_com_pii directly
                    texto_gerado_com_pii = generated_data.get('texto_gerado_com_pii', '')

                    if not texto_gerado_com_pii:
                        logger.warning(f'Empty texto_gerado_com_pii (attempt {retry_count + 1})')
                        retry_count += 1
                        continue

                    logger.info(f'Valid JSON response found on attempt {retry_count + 1}')
                    preview = (
                        texto_gerado_com_pii[:100] + '...' if len(texto_gerado_com_pii) > 100 else texto_gerado_com_pii
                    )
                    logger.info(f'Text preview ({len(texto_gerado_com_pii)} chars): {preview}')

                    # Create masked text
                    texto_com_mascara = create_masked_text(texto_gerado_com_pii, pii_values_dict)
                    if texto_com_mascara is None:
                        logger.warning(
                            f'Masking failed for attempt {retry_count + 1}, discarding sample and trying again'
                        )
                        retry_count += 1
                        continue

                    # Validate masking was successful with word boundary checks
                    masking_successful = True
                    unmasked_pii = []
                    if texto_gerado_com_pii and texto_com_mascara:
                        for pii_type, pii_value in pii_values_dict.items():
                            if pii_value and isinstance(pii_value, str):
                                # For city_uf and other short codes, use word boundary check
                                if pii_type == 'city_uf' and len(pii_value) <= 2:
                                    # Check with word boundaries
                                    import re

                                    if re.search(rf'\b{re.escape(pii_value)}\b', texto_com_mascara, re.IGNORECASE):
                                        logger.warning(
                                            f"Attempt {retry_count + 1}: Masking validation failed! Original PII '{pii_value}' "
                                            f'(Type: {pii_type}) still found in masked text as a whole word.'
                                        )
                                        masking_successful = False
                                        unmasked_pii.append(f"{pii_type}: '{pii_value}'")
                                # For other PII types, use simple substring check
                                elif pii_value in texto_com_mascara:
                                    logger.warning(
                                        f"Attempt {retry_count + 1}: Masking validation failed! Original PII '{pii_value}' "
                                        f'(Type: {pii_type}) still found in masked text.'
                                    )
                                    masking_successful = False
                                    unmasked_pii.append(f"{pii_type}: '{pii_value}'")

                    if not masking_successful:
                        async with _rejected_sample_lock:
                            global _rejected_sample_counter
                            _rejected_sample_counter += 1
                            rejection_count = _rejected_sample_counter
                        # NEW: Display total stats
                        async with _generated_sample_lock:
                            global _generated_sample_counter
                            generated_count = _generated_sample_counter
                        logger.error(
                            f'Attempt {retry_count + 1}: Discarding result due to failed masking validation. '
                            f'Unmasked PII: {unmasked_pii} | Total rejected: {rejection_count} | Total generated: {generated_count}'
                        )
                        logger.error(
                            f'REJECTED SAMPLE #{rejection_count} - Reason: Masking validation failed, Unmasked PII: {unmasked_pii}, Generated text: {texto_gerado_com_pii[:200]}'
                        )
                        retry_count += 1
                        continue

                    logger.info(f'Attempt {retry_count + 1}: Masking validation passed.')
                    return texto_gerado_com_pii, texto_com_mascara

                except json.JSONDecodeError as e:
                    logger.warning(f'Failed to parse JSON response: {e} (attempt {retry_count + 1})')
                    retry_count += 1
                    continue

        except Exception as e:
            logger.exception(f'Attempt {retry_count + 1} failed: {e}')
            retry_count += 1
            await asyncio.sleep(min(backoff_factor * (2**retry_count), 8))

    msg = 'No valid content found in model response'
    raise ValueError(msg)


async def use_model_to_generate_PII(
    session: aiohttp.ClientSession, prompt: str, pii_values_dict: dict[str, str]
) -> tuple[str, str, str]:
    """Generate PII text using Qwen model with llguidance."""
    for retry_count in range(10):
        try:
            # Always use Qwen model with llguidance
            texto_gerado_com_pii, texto_com_mascara = await qwen_gerador_PII(
                session, prompt, pii_values_dict, retry_count
            )  # Pass session here
            model_used = DEFAULT_MODEL_ID
            return texto_gerado_com_pii, texto_com_mascara, model_used
        except Exception as e:
            logger.exception(f'Qwen generation attempt {retry_count + 1} failed: {e}')
    msg = 'Failed to generate after 10 attempts'
    raise ValueError(msg)


def main(
    count: int = typer.Option(None, '--count', '-c', help='Number of PII samples to generate'),
    output_file: str = typer.Option(None, '--output', '-o', help='Output file path (.jsonl)'),
    should_pollute: bool = typer.Option(False, '--pollute', '-p', help='Whether to apply data pollution'),
    pollution_rate: float = typer.Option(None, '--rate', '-r', help='Percentage of samples to pollute (0.0 to 1.0)'),
    primary_llm: str = typer.Option(None, '--primary-llm', '-l', help='Primary LLM model to use for generation'),
    repeated_pii: bool = typer.Option(
        False, '--repeated-pii', '-pr', help='Whether to include repeated PII pollution'
    ),
    ocr_spaces: bool = typer.Option(False, '--ocr-spaces', '-ps', help='Whether to include OCR space errors'),
    ocr_chars: bool = typer.Option(False, '--ocr-chars', '-pc', help='Whether to include OCR character errors'),
    encoding: bool = typer.Option(False, '--encoding', '-pe', help='Whether to include encoding issues'),
    numbers: bool = typer.Option(False, '--numbers', '-pn', help='Whether to include number errors'),
    formats: bool = typer.Option(False, '--formats', '-pf', help='Whether to include mixed format issues'),
    samples_source: str | None = typer.Option(
        DEFAULT_SAMPLE_SOURCE,
        '--sample-source',
        '-i',
        help='Path to JSONL file with sample data (default: input/samples.jsonl). If "none", generate data from scratch.',
    ),
    num_workers: int = typer.Option(
        DEFAULT_NUM_WORKERS, '--workers', '-w', help='Number of concurrent workers for generation'
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, '--batch-size', '-b', help='Number of results to save in each batch'
    ),
    easy: int | None = typer.Option(
        None,
        '--easy',
        '-e',
        help=f'Shortcut mode: Generate specified number of samples using input/samples.jsonl with default settings (output to outputs/generated_samples.jsonl, input to input/samples.jsonl, pollution enabled at {DEFAULT_POLLUTION_RATE * 100}%, {DEFAULT_NUM_WORKERS} workers), generating with {DEFAULT_MODEL_ID} model.',
    ),
) -> None:
    """Generate synthetic PII data with various pollution options.

    This command generates samples of text containing Personally Identifiable Information (PII),
    with options to apply realistic data pollution to simulate real-world data quality issues.

    Uses Google GenAI with the gemini-2.0-flash model for text generation.

    You can either generate PII data from scratch using the model, or use sample data from a JSONL file.
    Results are saved incrementally in batches for better reliability.

    Exit codes:
      0: Success - All requested samples were generated
      1: Error - Failed to start or complete generation
      2: Partial failure - Less than 50% of requested samples were generated
      130: User interrupted the operation
    """

    def signal_handler(signum, frame) -> NoReturn:
        logger.warning(f'Received signal {signum}, initiating graceful shutdown...')
        # Raising KeyboardInterrupt is a clean way to break the asyncio loop
        # as we already have a handler for it.
        raise KeyboardInterrupt

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # For Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # For termination signals
    pollution_types_list = []
    if easy is not None:
        count = easy  # Use the provided count parameter
        output_file = DEFAULT_OUTPUT_FILE
        should_pollute = DEFAULT_POLLUTE  # TODO add pollutor  # Always enable pollution in easy mode
        pollution_rate = DEFAULT_POLLUTION_RATE  # Force 100% pollution rate in easy mode
        sample_source = DEFAULT_SAMPLE_SOURCE
        num_workers = DEFAULT_NUM_WORKERS
        batch_size = DEFAULT_BATCH_SIZE
        primary_llm = DEFAULT_MODEL_ID  # TODO implement
        temperature = DEFAULT_TEMPERATURE  # TODO implement

        # Enable all pollution types in easy mode
        repeated_pii = True
        ocr_spaces = True
        ocr_chars = True
        encoding = True
        numbers = True
        formats = True

        logger.debug('Easy mode enabled: forcing pollution on all samples with all pollution types')

        # Ensure the output directory exists
        output_dir = Path(output_file).parent  # Use Path
        if output_dir and not output_dir.exists():  # Use Path.exists()
            output_dir.mkdir(parents=True, exist_ok=True)  # Use Path.mkdir()
            logger.info(f'Created output directory: {output_dir}')

        # Display a rich panel with the configuration if available
        if RICH_AVAILABLE and console:
            console.print(
                Panel.fit(
                    f'[bold green]Easy Mode Configuration[/]\n\n'
                    f'[bold]Count:[/] {count} samples\n'
                    f'[bold]Output:[/] outputs/generated_samples.jsonl \n'
                    f'[bold]Pollution:[/] Enabled ({pollution_rate * 100}% rate)\n'
                    f'[bold]Sample source:[/] input/samples.jsonl\n'
                    f'[bold]Workers:[/] {num_workers}\n'
                    f'[bold]Batch size:[/] {batch_size}\n'
                    f'[bold]Primary LLM:[/] {primary_llm}\n'
                    f'[bold]Temperature:[/] {temperature}\n'
                    f'[bold]Repeated PII:[/] {repeated_pii}\n'
                    f'[bold]OCR Spaces:[/] {ocr_spaces}\n'
                    f'[bold]OCR Chars:[/] {ocr_chars}\n'
                    f'[bold]Batch size:[/] {batch_size}',
                    title='PII Generator',
                    border_style='green',
                    padding=(1, 2),
                )
            )
        else:
            typer.echo(f'Easy mode: Generating {count} samples with default settings:')
            typer.echo(f'  - Output: {output_file}')
            typer.echo(f'  - Pollution: Enabled ({pollution_rate * 100}% rate)')
            typer.echo(f'  - Sample source: {sample_source}')
            typer.echo(f'  - Workers: {num_workers}')
            typer.echo(f'  - Batch size: {batch_size}')
            typer.echo('  - Using GEMINI_API_KEY from environment')

    if count is None or count < 1:
        typer.echo('Error: Count must be at least 1')
        raise typer.Exit(code=1)
    if not (0.0 <= pollution_rate <= 1.0):
        typer.echo('Error: Pollution rate must be between 0.0 and 1.0')
        raise typer.Exit(code=1)
    if num_workers < 1:
        typer.echo('Error: Number of workers must be at least 1')
        raise typer.Exit(code=1)

    if should_pollute:
        if repeated_pii:
            pollution_types_list.append(PollutionType.REPEATED_PII)
        if ocr_spaces:
            pollution_types_list.append(PollutionType.OCR_SPACES)
        if ocr_chars:
            pollution_types_list.append(PollutionType.OCR_CHARS)
        if encoding:
            pollution_types_list.append(PollutionType.ENCODING_ISSUES)
        if numbers:
            pollution_types_list.append(PollutionType.NUMBER_ERRORS)
        if formats:
            pollution_types_list.append(PollutionType.MIXED_FORMATS)

    samples_source = None
    if sample_source != 'none':
        try:
            typer.echo(f'Loading samples from {sample_source}...')
            samples_source = asyncio.run(load_samples_from_jsonl(sample_source))
            typer.echo(f'Loaded {len(samples_source)} samples')
        except KeyboardInterrupt:
            if RICH_AVAILABLE and console:
                console.print('[bold yellow]Operation cancelled by user - results saved up to interruption point[/]')
            else:
                typer.echo('\nOperation cancelled by user - results saved up to interruption point')
            raise typer.Exit(code=130) from None
        except typer.Exit:
            # Re-raise typer.Exit exceptions to preserve their exit codes
            raise
        except Exception as e:
            typer.echo(f'Error loading samples: {e}')
            if sample_source != DEFAULT_SAMPLE_SOURCE:
                raise typer.Exit(code=1) from e  # Add 'from e'
            typer.echo('Falling back to model-based generation')
            samples_source = None

    typer.echo(f'Generating {count} PII samples using {num_workers} workers...')

    if easy is not None and should_pollute:
        pollution_types_list = [
            PollutionType.REPEATED_PII,
            PollutionType.OCR_SPACES,
            PollutionType.OCR_CHARS,
            PollutionType.ENCODING_ISSUES,
            PollutionType.NUMBER_ERRORS,
            PollutionType.MIXED_FORMATS,
        ]

    try:
        asyncio.run(
            process_with_workers(
                count=count,
                output_file=output_file,
                should_pollute=should_pollute,
                pollution_rate=pollution_rate,
                pollution_types=pollution_types_list,
                samples_source=samples_source,
                num_workers=num_workers,
                batch_size=batch_size,
            )
        )

    except KeyboardInterrupt:
        if RICH_AVAILABLE and console:
            console.print('[bold red]Operation cancelled by user[/]')
        else:
            typer.echo('\nOperation cancelled by user')
        raise typer.Exit(code=130) from None  # Add 'from None'
    except typer.Exit:
        # Re-raise typer.Exit exceptions to preserve their exit codes
        raise
    except Exception as e:
        if RICH_AVAILABLE and console:
            console.print(f'[bold red]Error:[/] {e}')
            # Print the traceback with rich formatting
            import traceback

            console.print_exception()
        else:
            typer.echo(f'Error: {e}')
            traceback.print_exc()
        raise typer.Exit(code=1) from e  # Add 'from e'


if __name__ == '__main__':
    # Rich console is already set up at the top of the file
    if RICH_AVAILABLE:
        console.print(
            Panel.fit(
                '[bold green]PII Generator[/bold green] - Generate synthetic PII data with optional pollution',
                border_style='green',
                padding=(1, 2),
            )
        )
    typer.run(main)
