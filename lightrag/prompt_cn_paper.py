from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

# PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["research_field", "theory_model", "experimental_method", "dataset", "metric", "academic_org", "scholar", "venue", "research_problem", "dataset", "method", "resource", "case_study", "evaluation", "technology", "framework", "concept"]

PROMPTS["entity_extraction"] = """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [dataset, method, research_problem, resource]
Text:
针对目前缺少面向表演的数据支撑及建模研究、生成过程缺乏对创意规则的挖掘等问题，本文首先通过网络资源收集了约 11.4GB 的表演创意相关文本语料，构成中国表演创意文本数据集；然后在此基础上对优秀的文本语料进行面向表演创意的主题建模，深入研究了创意实体分布规则. 通过对创意实体的分布规律进行挖掘，为表演创意的生成提供了数据支撑.
################
Output:
("entity"{tuple_delimiter}"中国表演创意文本数据集"{tuple_delimiter}"dataset"{tuple_delimiter}"通过收集11.4GB表演创意相关文本语料构建的专项数据集，用于支撑表演创意建模研究"){record_delimiter}
("entity"{tuple_delimiter}"主题建模"{tuple_delimiter}"method"{tuple_delimiter}"应用于优秀文本语料的统计分析方法，用于挖掘表演创意主题分布特征"){record_delimiter}
("entity"{tuple_delimiter}"创意实体分布规则"{tuple_delimiter}"research_problem"{tuple_delimiter}"研究表演创意文本中实体分布的统计规律，解决生成过程规则缺失问题"){record_delimiter}
("entity"{tuple_delimiter}"网络资源"{tuple_delimiter}"resource"{tuple_delimiter}"用于采集原始文本语料的互联网数据来源渠道"){record_delimiter}
("entity"{tuple_delimiter}"创意实体分布规律"{tuple_delimiter}"research_problem"{tuple_delimiter}"通过数据挖掘揭示的创意实体分布模式，为生成算法提供规则依据"){record_delimiter}
("relationship"{tuple_delimiter}"网络资源"{tuple_delimiter}"中国表演创意文本数据集"{tuple_delimiter}"网络资源作为数据采集来源支撑数据集构建"{tuple_delimiter}"数据采集,资源转化"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"中国表演创意文本数据集"{tuple_delimiter}"主题建模"{tuple_delimiter}"数据集为主题建模方法提供分析基础"{tuple_delimiter}"数据驱动,方法应用"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"主题建模"{tuple_delimiter}"创意实体分布规则"{tuple_delimiter}"通过主题建模方法发现实体分布规律"{tuple_delimiter}"模式发现,规则提取"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"创意实体分布规则"{tuple_delimiter}"创意实体分布规律"{tuple_delimiter}"规则研究推导出可量化的分布模式"{tuple_delimiter}"理论转化,应用延伸"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"创意实体分布规律"{tuple_delimiter}"表演创意的生成"{tuple_delimiter}"分布规律为创意生成提供数据支撑"{tuple_delimiter}"规则嵌入,生成优化"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"表演艺术数字化,文本挖掘,数据驱动创新,生成式规则建模"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [research_problem, method, resource, case_study, evaluation]
Text:
针对创意方法中难以保持创意元素情感一致的问题，本文在创意实体分布规则和创意资源知识库的基础上，提出多约束条件下智能化创意生成方法，并以此为基础进行了“雪花”、“立春”为主题，不同扩展词数的创意生成和对比评估。由于创意一定程度上的主观性，采用了主客观双视角共六个指标的方式对提出的方法进行多维度实验评估.
#############
Output:
("entity"{tuple_delimiter}"多约束条件下智能化创意生成方法"{tuple_delimiter}"method"{tuple_delimiter}"基于创意实体分布规则与知识库提出的生成算法，用于解决创意元素情感一致性保持问题"){record_delimiter}
("entity"{tuple_delimiter}"创意实体分布规则"{tuple_delimiter}"research_problem"{tuple_delimiter}"作为方法基础之一的创意元素统计规律研究"){record_delimiter}
("entity"{tuple_delimiter}"创意资源知识库"{tuple_delimiter}"resource"{tuple_delimiter}"存储结构化创意元素的数据库，为生成方法提供素材支撑"){record_delimiter}
("entity"{tuple_delimiter}"雪花"{tuple_delimiter}"case_study"{tuple_delimiter}"作为生成方法验证的主题案例之一"){record_delimiter}
("entity"{tuple_delimiter}"立春"{tuple_delimiter}"case_study"{tuple_delimiter}"作为生成方法验证的对比主题案例"){record_delimiter}
("entity"{tuple_delimiter}"主客观双视角评估指标"{tuple_delimiter}"evaluation"{tuple_delimiter}"包含6个维度的综合评估体系，兼顾主观审美与客观量化分析"){record_delimiter}
("relationship"{tuple_delimiter}"创意实体分布规则"{tuple_delimiter}"多约束条件下智能化创意生成方法"{tuple_delimiter}"分布规则为生成方法提供理论约束"{tuple_delimiter}"规则嵌入,方法设计"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"创意资源知识库"{tuple_delimiter}"多约束条件下智能化创意生成方法"{tuple_delimiter}"知识库为方法提供数据素材支持"{tuple_delimiter}"数据支撑,资源调用"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"多约束条件下智能化创意生成方法"{tuple_delimiter}"雪花"{tuple_delimiter}"方法应用于该主题的创意生成"{tuple_delimiter}"案例验证,应用测试"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"多约束条件下智能化创意生成方法"{tuple_delimiter}"立春"{tuple_delimiter}"方法应用于对比主题的生成"{tuple_delimiter}"对比实验,效果评估"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"主客观双视角评估指标"{tuple_delimiter}"多约束条件下智能化创意生成方法"{tuple_delimiter}"通过多维度指标验证方法有效性"{tuple_delimiter}"综合验证,效果度量"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"创意一致性保持,多约束生成,主客观评估,智能创作系统"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [technology, method, resource, dataset, framework, concept, evaluation]
Text:
本文首先基于实体联想和语义分割算法构建表演创意元素知识库。围绕创意中心主题词汇，基于知识图谱技术建立实体语义网络，通过实体联想算法，确定相关视觉实体对象，通过赋予计算机以人类智能的联想能力，实现范围更广泛和分类更清晰的创意联想。再通过语义搜索获得创意主体的视觉素材，提取前景对象构建数据集，训练语义类别分割模型，实现用于对相应语义类别对象的自动提取分割，构建具有中国文化特色的表演创意对象素材库，为创意画面合成提供大量的、丰富的、高质的实体对象视觉素材。在获得创意素材后，基于图像合成算法进行表演创意画面可视化，算法分为物体放置，图像融合及图像和谐化三个部分，首先选取恰当的前景对象和背景，编写算法进行基于美学原则的合理构图，基于泊松融合进行边界的无缝融合，并对光照纹理进行和谐处理，使画面更真实美观。最后从美观性、合理性和创意性三个维度对生成画面进行评估，构建多维度评估体系，为创意画面生成提供评价标准。
#############
Output:
("entity"{tuple_delimiter}"实体联想算法"{tuple_delimiter}"method"{tuple_delimiter}"模拟人类联想能力实现创意实体关联的算法"){record_delimiter}
("entity"{tuple_delimiter}"语义分割模型"{tuple_delimiter}"method"{tuple_delimiter}"用于提取前景对象并构建素材库的深度学习模型"){record_delimiter}
("entity"{tuple_delimiter}"表演创意元素知识库"{tuple_delimiter}"resource"{tuple_delimiter}"基于知识图谱技术构建的结构化创意元素数据库"){record_delimiter}
("entity"{tuple_delimiter}"语义类别分割数据集"{tuple_delimiter}"dataset"{tuple_delimiter}"通过语义搜索获取并标注的视觉素材训练集"){record_delimiter}
("entity"{tuple_delimiter}"图像合成算法"{tuple_delimiter}"method"{tuple_delimiter}"包含物体放置、融合与和谐化的三阶段画面生成算法"){record_delimiter}
("entity"{tuple_delimiter}"泊松融合"{tuple_delimiter}"method"{tuple_delimiter}"实现图像边界无缝融合的数学方法"){record_delimiter}
("entity"{tuple_delimiter}"中国文化特色素材库"{tuple_delimiter}"resource"{tuple_delimiter}"包含分类清晰的高质量表演视觉素材库"){record_delimiter}
("entity"{tuple_delimiter}"知识图谱技术"{tuple_delimiter}"technology"{tuple_delimiter}"用于构建结构化语义网络的核心技术，支持实体关系可视化与推理"){record_delimiter}
("entity"{tuple_delimiter}"实体语义网络"{tuple_delimiter}"framework"{tuple_delimiter}"基于知识图谱的语义关联框架，实现创意元素的多维连接"){record_delimiter}
("entity"{tuple_delimiter}"创意中心主题词汇"{tuple_delimiter}"concept"{tuple_delimiter}"指导知识库构建的核心语义主题集合"){record_delimiter}
("entity"{tuple_delimiter}"物体放置算法"{tuple_delimiter}"method"{tuple_delimiter}"图像合成中基于美学原则的空间布局优化方法"){record_delimiter}
("entity"{tuple_delimiter}"图像融合技术"{tuple_delimiter}"method"{tuple_delimiter}"包含泊松融合等多种方法的视觉合成技术大类"){record_delimiter}
("entity"{tuple_delimiter}"多维度评估体系"{tuple_delimiter}"evaluation"{tuple_delimiter}"包含美观性（视觉协调）、合理性（逻辑自洽）、创意性（新颖程度）的三维评价框架"){record_delimiter}
("entity"{tuple_delimiter}"美观性评估"{tuple_delimiter}"evaluation"{tuple_delimiter}"衡量视觉协调性的指标")  
("entity"{tuple_delimiter}"合理性评估"{tuple_delimiter}"evaluation"{tuple_delimiter}"评估逻辑自洽性的指标"){record_delimiter}
("entity"{tuple_delimiter}"创意性评估"{tuple_delimiter}"evaluation"{tuple_delimiter}"量化创新程度的指标"){record_delimiter}
("relationship"{tuple_delimiter}"实体联想算法"{tuple_delimiter}"表演创意元素知识库"{tuple_delimiter}"算法驱动知识库实体网络构建"{tuple_delimiter}"知识建模,算法驱动"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"语义分割模型"{tuple_delimiter}"中国文化特色素材库"{tuple_delimiter}"模型实现素材自动提取与分类"{tuple_delimiter}"数据处理,素材生成"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"语义类别分割数据集"{tuple_delimiter}"语义分割模型"{tuple_delimiter}"数据集支撑模型训练"{tuple_delimiter}"数据驱动,模型优化"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"图像合成算法"{tuple_delimiter}"泊松融合"{tuple_delimiter}"融合技术作为合成算法的核心组件"{tuple_delimiter}"技术集成,流程优化"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"中国文化特色素材库"{tuple_delimiter}"图像合成算法"{tuple_delimiter}"素材库为合成算法提供视觉元素"{tuple_delimiter}"资源供给,创意实现"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"知识图谱技术"{tuple_delimiter}"实体语义网络"{tuple_delimiter}"技术支撑网络架构建设"{tuple_delimiter}"技术实现,框架构建"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"创意中心主题词汇"{tuple_delimiter}"实体语义网络"{tuple_delimiter}"主题词汇驱动语义网络构建"{tuple_delimiter}"语义引导,知识建模"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"图像合成算法"{tuple_delimiter}"物体放置算法"{tuple_delimiter}"合成流程包含空间布局优化阶段"{tuple_delimiter}"流程分解,技术集成"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"图像融合技术"{tuple_delimiter}"泊松融合"{tuple_delimiter}"泊松融合属于图像融合技术子类"{tuple_delimiter}"技术分层,方法继承"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"多维度评估体系"{tuple_delimiter}"美观性评估"{tuple_delimiter}"体系包含视觉协调性评估维度"{tuple_delimiter}"评估架构,指标分解"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"多维度评估体系"{tuple_delimiter}"合理性评估"{tuple_delimiter}"体系包含逻辑自洽性评估维度"{tuple_delimiter}"评估架构,指标分解"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"多维度评估体系"{tuple_delimiter}"创意性评估"{tuple_delimiter}"体系包含创新性评估维度"{tuple_delimiter}"评估架构,指标分解"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"知识图谱驱动,跨模态融合,智能美学计算,文化数字化,多维评估体系,语义增强创作"){completion_delimiter}
#############################""",
    """Example 4:

Entity_types: [technology, method, resource, dataset]
Text:
本文首先基于实体联想和语义分割算法构建表演创意元素知识库。围绕创意中心主题词汇，基于知识图谱技术建立实体语义网络，通过实体联想算法，确定相关视觉实体对象，通过赋予计算机以人类智能的联想能力，实现范围更广泛和分类更清晰的创意联想。再通过语义搜索获得创意主体的视觉素材，提取前景对象构建数据集，训练语义类别分割模型，实现用于对相应语义类别对象的自动提取分割，构建具有中国文化特色的表演创意对象素材库，为创意画面合成提供大量的、丰富的、高质的实体对象视觉素材。在获得创意素材后，基于图像合成算法进行表演创意画面可视化，算法分为物体放置，图像融合及图像和谐化三个部分，首先选取恰当的前景对象和背景，编写算法进行基于美学原则的合理构图，基于泊松融合进行边界的无缝融合，并对光照纹理进行和谐处理，使画面更真实美观。
#############
Output:
("entity"{tuple_delimiter}"知识图谱技术"{tuple_delimiter}"technology"{tuple_delimiter}"用于构建结构化语义网络的核心技术，支持实体关系可视化与推理"){record_delimiter}
("entity"{tuple_delimiter}"实体语义网络"{tuple_delimiter}"method"{tuple_delimiter}"基于知识图谱的语义关联框架，实现创意元素的多维连接"){record_delimiter}
("entity"{tuple_delimiter}"实体联想算法"{tuple_delimiter}"method"{tuple_delimiter}"模拟人类联想能力实现创意实体关联的算法"){record_delimiter}
("entity"{tuple_delimiter}"语义分割模型"{tuple_delimiter}"method"{tuple_delimiter}"用于提取前景对象并构建素材库的深度学习模型"){record_delimiter}
("entity"{tuple_delimiter}"表演创意元素知识库"{tuple_delimiter}"resource"{tuple_delimiter}"基于知识图谱技术构建的结构化创意元素数据库"){record_delimiter}
("entity"{tuple_delimiter}"语义类别分割数据集"{tuple_delimiter}"dataset"{tuple_delimiter}"通过语义搜索获取并标注的视觉素材训练集"){record_delimiter}
("entity"{tuple_delimiter}"图像合成算法"{tuple_delimiter}"method"{tuple_delimiter}"包含物体放置、融合与和谐化的三阶段画面生成算法"){record_delimiter}
("entity"{tuple_delimiter}"泊松融合"{tuple_delimiter}"method"{tuple_delimiter}"实现图像边界无缝融合的数学方法"){record_delimiter}
("entity"{tuple_delimiter}"中国文化特色素材库"{tuple_delimiter}"resource"{tuple_delimiter}"包含分类清晰的高质量表演视觉素材库"){record_delimiter}
("relationship"{tuple_delimiter}"实体联想算法"{tuple_delimiter}"表演创意元素知识库"{tuple_delimiter}"算法驱动知识库实体网络构建"{tuple_delimiter}"知识建模,算法驱动"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"语义分割模型"{tuple_delimiter}"中国文化特色素材库"{tuple_delimiter}"模型实现素材自动提取与分类"{tuple_delimiter}"数据处理,素材生成"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"语义类别分割数据集"{tuple_delimiter}"语义分割模型"{tuple_delimiter}"数据集支撑模型训练"{tuple_delimiter}"数据驱动,模型优化"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"图像合成算法"{tuple_delimiter}"泊松融合"{tuple_delimiter}"融合技术作为合成算法的核心组件"{tuple_delimiter}"技术集成,流程优化"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"中国文化特色素材库"{tuple_delimiter}"图像合成算法"{tuple_delimiter}"素材库为合成算法提供视觉元素"{tuple_delimiter}"资源供给,创意实现"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"智能知识建模,跨模态融合,文化数字化,生成式视觉合成"){completion_delimiter}
#############################""",
    """Example 5:

Entity_types: [case_study, method, dataset, resource, evaluation]
Text:
本文以北京 2022 年冬季奥运会开幕式创意主题“立春”为关键词进行实验，对选自中国权威辞典及文献总计约八万字的数据资料进行实体识别、关系提取，建立了一共包含 11700 张创意素材的知识库，并按照奥运会开幕式的画面设计添加字体等素材形成最终的创意画面。最后从美观性、合理性和创意性三个维度与其他实验方法包括语义搜索方法，图像生成方法，从定量和定性的角度进行对比评估，实验数据证明本文方法在美观性、创意性和合理性上均优于其他方法。
#############
Output:
("entity"{tuple_delimiter}"北京2022年冬季奥运会开幕式"{tuple_delimiter}"case_study"{tuple_delimiter}"以'立春'为主题的冬奥会开幕式创意设计案例"){record_delimiter}
("entity"{tuple_delimiter}"立春"{tuple_delimiter}"case_study"{tuple_delimiter}"冬奥会开幕式的核心创意主题关键词"){record_delimiter}
("entity"{tuple_delimiter}"中国权威辞典及文献数据集"{tuple_delimiter}"dataset"{tuple_delimiter}"约八万字的中文语料数据源，用于实体识别与关系抽取"){record_delimiter}
("entity"{tuple_delimiter}"创意素材知识库"{tuple_delimiter}"resource"{tuple_delimiter}"包含11700张素材的结构化资源库，含字体等画面设计元素"){record_delimiter}
("entity"{tuple_delimiter}"多维度评估体系"{tuple_delimiter}"evaluation"{tuple_delimiter}"从美观性、合理性和创意性三个维度构建的主客观评价框架"){record_delimiter}
("entity"{tuple_delimiter}"语义搜索方法"{tuple_delimiter}"method"{tuple_delimiter}"作为对比实验基准的传统检索方法"){record_delimiter}
("entity"{tuple_delimiter}"图像生成方法"{tuple_delimiter}"method"{tuple_delimiter}"作为对比实验的自动化生成技术"){record_delimiter}
("relationship"{tuple_delimiter}"立春"{tuple_delimiter}"创意素材知识库"{tuple_delimiter}"主题关键词驱动知识库构建"{tuple_delimiter}"主题驱动,资源整合"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"中国权威辞典及文献数据集"{tuple_delimiter}"实体识别方法"{tuple_delimiter}"数据集支撑实体抽取流程"{tuple_delimiter}"数据支撑,知识发现"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"创意素材知识库"{tuple_delimiter}"图像合成方法"{tuple_delimiter}"知识库提供画面合成素材"{tuple_delimiter}"资源供给,创作实现"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"多维度评估体系"{tuple_delimiter}"本文方法"{tuple_delimiter}"通过三维度指标验证方法优越性"{tuple_delimiter}"综合验证,效果度量"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"语义搜索方法"{tuple_delimiter}"图像生成方法"{tuple_delimiter}"作为对比实验的基准方法组"{tuple_delimiter}"方法对比,效果参照"{tuple_delimiter}6){record_delimiter}
("content_keywords"{tuple_delimiter}"奥运数字人文,创意知识工程,跨模态评估,文化科技融合"){completion_delimiter}
"""
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Base provided below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base."""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks."""


PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["mix_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sesctions focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" sesction. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
