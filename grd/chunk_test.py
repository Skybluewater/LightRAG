from chonkie import LateChunker, SemanticChunker
from siliconflow_embeddings import SiliconFlowEmbeddings
from chonkie import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

# import os
# os.environ["HF_HOME"] = "E:\\huggingface"

model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

task = "retrieval.query"
embeddings = model.encode(
    ["What is the weather like in Berlin today?"],
    task=task,
    prompt_name=task,
)

chunker_semantic = SemanticChunker(
    embedding_model=SiliconFlowEmbeddings(),
    threshold="auto",
    chunk_size=512
)

text = """创意是表演艺术中最为重要和令人兴奋的元素之一。如今的表演形式日益多样化，而表演作品的质量往往取决于背后的创意水平。传统的表演创意过程以导演的经验为主导，很大程度上围绕导演的艺术素养和风格导向展开[1]，但这种方式往往需要耗费大量时间和资金，同时也容易受到导演审美疲劳的影响，导致创意思路受限、创意效果不佳等问题。随着新一代信息技术的发展，数字化方法为解决这些问题带来新思路和新视野上的拓展。计算机技术的应用为表演创意智能化提供了更多可能性和更广阔的前景。
为了能够让表演创意过程更高效、创意效果更精彩，人工智能、情感计算等前沿技术被用来跨学科研究。例如：通过对表演中关键创意元素进行数字化描述形成基于表演创意的智能涌现[2]；利用人机交互和情感计算进行基于观众关注度的表演创意评估[3]；通过建立虚拟仿真表演空间来对创意效果进行实景推演[4]等。这些研究利用数字化和智能化的方法在如何产生和改进表演创意的过程中取得了一定的进展，但是面对愈发复杂的表演环境、愈发多样的表演内容以及观众对表演艺术的审美和认知的不断提高，如何提升表演创意生成的效率和质量还是一个十分具有挑战性的问题。因为在面向表演进行创意生成时，需要根据表演主题、传达的情感、观众群体等因素来生成相应的剧本，这就需要对现有的剧本、文化、情感载体等资料进行深入学习。然而单纯的模仿还难以满足要求，因为如何保持创意性是第二大难题，只有具有一定的创造力和想象力，才能生成有趣又独特的内容。这就需要模型既有一定的学习能力和模仿能力，又有创造力，能够自主提出新的内容。除此之外，为了避免生成重复和低质量内容，还需要模型具有一定的判断能力和筛选能力，才能自动区分创意结果的好坏、新旧、有趣无趣等。随着互联网时代信息的爆炸式增长，大量的数据积累为发现和分析现有规律奠定基础，同时也为机器的学习和模仿提供参照物，但如果机器仅仅是在数据规律下进行仿写或续写，会导致完成的“创意”缺乏情感色彩和共情表达能力。由此我们提出了面向规则研究和情感加成的创意生成框架，进行基于情感的表演创意生成。具体包括：基于优秀剧本的中国表演创意建模，基于情感权重的创意资源知识
近年来，随着我国经济的发展和国力的增强，在国际舞台上承办了多项重大体育赛事，如 2008 年北京夏季奥运会和 2022 年北京冬季奥运会等。同时，就国内而言也围绕着一系列标志性历史节点举行了不少庆典和主题演出，如庆祝中国共产党成立100 周年文艺演出大型情景史诗 “伟大征程”等。这些文艺演出作为庆典活动的主要形式之一，通过精心的编排创作，将我国博大精深且源远流长的文化搬上舞台。文艺表演不仅能够为庆典活动增添节日氛围，更重要的是可以通过舞台叙事展现富有民族意识的象征和符号[5]，使观众更深刻地感受到表演的魅力，增强对身份的认同感和文化自豪感。表演的价值已经逐渐超过了活动自身，而是在为整个社会塑造着一个共同拥有的时代印记。因此举办更具有意义和深度的文艺表演对促进文化传播和价值交流至关重要。
受当下数字经济政策导向和互联网交叉领域成果影响，艺术表演领域的智能化研究也成为一大热点，但是面向中国表演创意元素构成的可计算知识体系构建方面稍显空白，我们习惯用理性的思维和通用的元素处理方法去描述事实，这种方式固然可以迁移到创意生成中去，却会导致艺术性质量低，生成创意重复度高等问题。智能创意生成方法真正将计算机智能引入艺术表演中去，突破表演创意体系缺乏、数字化进程慢、智能化程度低的问题，提高艺术创作工作效率，通过学习大量数据和模式来发掘更多的创意可能性，拓展创意边界，提高创意质量，推动人工智能技术在艺术领域的发展，让创意生成更好地服务于人类的生产和生活。目前也已经有很多成功的交叉应用都让我们看到创意智能化的发展和应用前景，但是针对表演创意生成这一领域的研究与开发还略显空白。"""

semantic_chunks = chunker_semantic(text)

for chunk in semantic_chunks:
    print(f"Chunk text: {chunk.text}")
    print(f"Token count: {chunk.token_count}")
    print(f"Number of sentences: {len(chunk.sentences)}")


late_chunker = LateChunker(
    embedding_model="jinaai/jina-embeddings-v3",
    mode = "sentence",
    chunk_size=512,
    min_sentences_per_chunk=1,
    min_characters_per_sentence=12,
    trust_remote_code=True
)

late_chunks = late_chunker(text)

for chunk in late_chunks:
    print(f"Chunk text: {chunk.text}")
    print(f"Token count: {chunk.token_count}")
    print(f"Number of sentences: {len(chunk.sentences)}")