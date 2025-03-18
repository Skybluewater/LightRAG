import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("PrunaAI/zjunlp-OneKE-bnb-4bit-smashed", trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("zjunlp/OneKE")

model.eval()
# input_ids = tokenizer("What is the color of prunes?,", return_tensors='pt').to(model.device)["input_ids"]

# outputs = model.generate(input_ids, max_new_tokens=216)
# tokenizer.decode(outputs[0])
system_prompt = '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n'
# sintruct = "{\"instruction\": \"You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.\",\"schema\": [\"technology\", \"method\", \"else\", \"idea\"], \"input\": \"文艺表演是社会重要的文化活动，不仅体现了社会的精神追求，更是有力的文化\n传播工具。表演创意是文艺表演的关键，是文化、思想、艺术的集中体现。表演视觉\n画面是表演创意的核心，创意通常被看作是人主观的创造活动，随着计算机算法智能\n水平不断提高，通过智能计算实现智能创意逐渐成为可能。而将智能计算应用到表演\n创意领域，实现智能创意表演尚处于发展初期。本文首先提出了基于实体联想的表演\n创意画面合成算法，实现表演创意的激发和可视化。 \n本文首先基于实体联想和语义分割算法构建表演创意元素知识库。围绕创意中心\n主题词汇，基于知识图谱技术建立实体语义网络，通过实体联想算法，确定相关视觉\n实体对象，通过赋予计算机以人类智能的联想能力，实现范围更广泛和分类更清晰的\n创意联想。再通过语义搜索获得创意主体的视觉素材，提取前景对象构建数据集，训\n练语义类别分割模型，实现用于对相应语义类别对象的自动提取分割，构建具有中国\n文化特色的表演创意对象素材库，为创意画面合成提供大量的、丰富的、高质的实体\n对象视觉素材。在获得创意素材后，基于图像合成算法进行表演创意画面可视化，算\n法分为物体放置，图像融合及图像和谐化三个部分，首先选取恰当的前景对象和背景，\n编写算法进行基于美学原则的合理构图，基于泊松融合进行边界的无缝融合，并对光\n照纹理进行和谐处理，使画面更真实美观。\"}"
sintruct = "{\"instruction\": \"你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。\",\"schema\": [\"技术\", \"实现\"], \"input\": \"文艺表演是社会重要的文化活动，不仅体现了社会的精神追求，更是有力的文化传播工具。表演创意是文艺表演的关键，是文化、思想、艺术的集中体现。表演视觉画面是表演创意的核心，创意通常被看作是人主观的创造活动，随着计算机算法智能水平不断提高，通过智能计算实现智能创意逐渐成为可能。而将智能计算应用到表演创意领域，实现智能创意表演尚处于发展初期。本文首先提出了基于实体联想的表演创意画面合成算法，实现表演创意的激发和可视化。 本文首先基于实体联想和语义分割算法构建表演创意元素知识库。围绕创意中心主题词汇，基于知识图谱技术建立实体语义网络，通过实体联想算法，确定相关视觉实体对象，通过赋予计算机以人类智能的联想能力，实现范围更广泛和分类更清晰的创意联想。再通过语义搜索获得创意主体的视觉素材，提取前景对象构建数据集，训练语义类别分割模型，实现用于对相应语义类别对象的自动提取分割，构建具有中国文化特色的表演创意对象素材库，为创意画面合成提供大量的、丰富的、高质的实体对象视觉素材。在获得创意素材后，基于图像合成算法进行表演创意画面可视化，算法分为物体放置，图像融合及图像和谐化三个部分，首先选取恰当的前景对象和背景，编写算法进行基于美学原则的合理构图，基于泊松融合进行边界的无缝融合，并对光照纹理进行和谐处理，使画面更真实美观。\"}"
sintruct = '[INST] ' + system_prompt + sintruct + '[/INST]'

input_ids = tokenizer.encode(sintruct, return_tensors="pt").to(device)
input_length = input_ids.size(1)
generation_output = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_length=1024, max_new_tokens=512, return_dict_in_generate=True), pad_token_id=tokenizer.eos_token_id)
generation_output = generation_output.sequences[0]
generation_output = generation_output[input_length:]
output = tokenizer.decode(generation_output, skip_special_tokens=True)

print(output)