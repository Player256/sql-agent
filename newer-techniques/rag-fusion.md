{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import boto3\n",
    "from langchain_aws import ChatBedrock\n",
    "from botocore.config import Config\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "region = \"us-west-2\"\n",
    "config = Config(\n",
    "    region_name=region,\n",
    "    signature_version = \"v4\",\n",
    "    retries={\n",
    "        \"max_attempts\":3,\n",
    "        \"mode\" : \"standard\",\n",
    "    }\n",
    ")\n",
    "bedrock_rt = boto3.client(\"bedrock-runtime\", config=config)\n",
    "\n",
    "sonnet_model_id = \"anthropic.claude-3-sonnet-20240229-v1:0\"\n",
    "\n",
    "model_kwargs = {\n",
    "    \"max_tokens\" : 4096,\n",
    "    \"temperature\" : 0.0,\n",
    "    \"stop_sequences\" : [\"Human\"],\n",
    "}\n",
    "\n",
    "llm = ChatBedrock(\n",
    "    client = bedrock_rt,\n",
    "    model_id = sonnet_model_id,\n",
    "    model_kwargs = model_kwargs,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_community.embeddings import BedrockEmbeddings\n",
    "\n",
    "bedrock_client = boto3.client(service_name='bedrock-runtime', \n",
    "                              region_name='us-east-1')\n",
    "embeddings_model = BedrockEmbeddings(model_id=\"amazon.titan-embed-text-v1\",\n",
    "                                       client=bedrock_client)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import getpass\n",
    "from langchain_cohere import CohereEmbeddings\n",
    "\n",
    "os.environ['COHERE_API_KEY'] = getpass.getpass()\n",
    "embeddings_model = CohereEmbeddings(\n",
    "    model=\"embed-english-light-v3.0\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**RAG Fusion**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "621\n"
     ]
    }
   ],
   "source": [
    "from langchain_community.document_loaders import PyPDFLoader\n",
    "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
    "from langchain_community.vectorstores import FAISS\n",
    "\n",
    "file_path = (\n",
    "    \"/home/ubuntu/learn/Insurance_Handbook_20103.pdf\"\n",
    ")\n",
    "loader = PyPDFLoader(file_path,extract_images = True)\n",
    "pages = loader.load_and_split()\n",
    "text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 0)\n",
    "docs = text_splitter.split_documents(pages)\n",
    "db = FAISS.from_documents(docs , embeddings_model)\n",
    "print(db.index.ntotal)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "================================\u001b[1m System Message \u001b[0m================================\n",
      "\n",
      "You are a helpful assistant that generates multiple search queries based on a single input query.\n",
      "\n",
      "================================\u001b[1m Human Message \u001b[0m=================================\n",
      "\n",
      "Generate multiple search queries related to: \u001b[33;1m\u001b[1;3m{original_query}\u001b[0m\n",
      "\n",
      "================================\u001b[1m Human Message \u001b[0m=================================\n",
      "\n",
      "OUTPUT (4 queries):\n"
     ]
    }
   ],
   "source": [
    "from langchain_core.output_parsers import StrOutputParser\n",
    "from langchain import hub\n",
    "\n",
    "prompt = hub.pull(\"langchain-ai/rag-fusion-query-generation\")\n",
    "prompt.pretty_print()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "original_query = \"How to identify in what scenarios I have to take an insurance?\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "generate_queries = (\n",
    "    prompt | llm | StrOutputParser() | (lambda x : x.split(\"\\n\"))\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "retriever = db.as_retriever()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.load import dumps, loads\n",
    "\n",
    "\n",
    "def reciprocal_rank_fusion(results: list[list], k=60):\n",
    "    fused_scores = {}\n",
    "    for docs in results:\n",
    "        # Assumes the docs are returned in sorted order of relevance\n",
    "        for rank, doc in enumerate(docs):\n",
    "            doc_str = dumps(doc)\n",
    "            if doc_str not in fused_scores:\n",
    "                fused_scores[doc_str] = 0\n",
    "            previous_score = fused_scores[doc_str]\n",
    "            fused_scores[doc_str] += 1 / (rank + k)\n",
    "\n",
    "    reranked_results = [\n",
    "        (loads(doc), score)\n",
    "        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)\n",
    "    ]\n",
    "    return reranked_results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "chain = generate_queries | retriever.map() | reciprocal_rank_fusion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(Document(metadata={'source': '/home/ubuntu/learn/Insurance_Handbook_20103.pdf', 'page': 199}, page_content='I.I.I.\\tInsurance\\tHandbook\\t\\t\\twww.iii.org/insurancehandbook\\t \\t193\\nI.I.I. \\nResources\\nI.I.I. Store\\nThe I.I.I. Store is your gateway to a wide array of books and brochures from the Insurance \\nInformation Institute. Print and PDF formats, and quantity discounts are available  for most products. Order online at www.iii.org/publications, call 212-346-5500 or email publications@iii.org.\\nI.I.I. INSURANCE FACT BOOK\\nThousands of insurance facts, figures, tables and graphs designed for quick and easy reference.\\nTHE FINANCIAL SERVICES FACT BOOK\\nBanking, securities and insurance industry trends and statistics. Published jointly with the Financial Services Roundtable. Online version available at www.financialservicesfacts.org\\nINSURANCE HANDBOOK'),\n",
       "  0.049206349206349205),\n",
       " (Document(metadata={'source': '/home/ubuntu/learn/Insurance_Handbook_20103.pdf', 'page': 84}, page_content='Furthermore, the Glossary is a compilation of definitions from various LOMA texts; however, it is not an assigned text for any LOMA \\ncourse. Sometimes a definition in the Glossary will differ somewhat from the definition in a text because of the nuances of the subject matter in the text. A student taking an exam always should rely on the definition in the assigned text rather than the one in the Glossary.88'),\n",
       "  0.03333333333333333),\n",
       " (Document(metadata={'source': '/home/ubuntu/learn/Insurance_Handbook_20103.pdf', 'page': 18}, page_content='Business Insurance'),\n",
       "  0.03333333333333333),\n",
       " (Document(metadata={'source': '/home/ubuntu/learn/Insurance_Handbook_20103.pdf', 'page': 67}, page_content='term or multiple years, often three.'),\n",
       "  0.03278688524590164),\n",
       " (Document(metadata={'source': '/home/ubuntu/learn/Insurance_Handbook_20103.pdf', 'page': 151}, page_content='SOCIETY OF ACTUARIES\\n475 North Martingale Road, Suite 600Schaumburg, IL 60173 Tel: 847-706-3500Fax: 847-706-3599Web: www.soa.orgAn educational, research and professional organization dedicated to serving the public and its members. The Society’s vision is for actuaries to be recognized as the leading professionals in the modeling and management of financial risk and contingent events.'),\n",
       "  0.03252247488101534)]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = chain.invoke({\"original_query\" : \"What are the contents of this pdf?\"})\n",
    "result[:5]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "pytorch",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
