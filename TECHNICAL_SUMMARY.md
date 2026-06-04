# Technical Summary — Upwork API Support Bots

  ## Difficulties faced & how I solved them

  Stopping hallucination without making the bot useless.
The hard requirement was "never invent an answer," but an over-strict prompt also refused questions the docs could answer (e.g. the Client Credentials question). I solved this by splitting the system prompt into two modes: factual-value questions (a specific number/limit/duration) must fall back if the value isn't literally present — which is why the rate-limit question correctly returns the fallback  while capability/permission questions are answered by reasoning over the documented behaviour. This kept the anti-hallucination guarantee and answeredthe questions the documentation supports.

  ## Chunking a messy PDF. `pypdf`
 extraction produced run-together words and inconsistent spacing. I used `RecursiveCharacterTextSplitter` (size 500, overlap 50) so chunks break on natural boundaries and the 50-char overlap keeps facts (like a token's `expires_in: 86400`) from being split across chunks and   lost at retrieval time.

## Embedding model reloading on every query.
 Because the flow is  `answer_question → retrieve → load_vectorstore`, the ~90 MB MiniLM model was being reloaded on *every* question. I memoised it with `@lru_cache` at the library level and `@st.cache_resource` in Streamlit, so it loads exactly once  turning multi-second per-query overhead into a one-time cost.

## API latency & reliability.
 DeepInfra calls vary in latency and can fail transiently. I measured the LLM call in isolation with `time.perf_counter()`, set `temperature=0` for deterministic answers, added a 60 s timeout, and wrapped
  the call in exponential-backoff retries with typed handling for 401/429/5xx/ timeout — every failure returns a friendly message, never a raw traceback.

 ## Windows environment quirks.
The console's legacy `cp1252` code page crashed on the Unicode characters in my status output, and ChromaDB's telemetry spammed  the terminal. I forced UTF-8 stdout and silenced the telemetry logger so the  project runs identically on Windows, macOS, and Linux.

 ## How I used LLMs to assist development

I used Claude as a pair-programmer, not an autopilot. It accelerated scaffolding the LangChain + ChromaDB + OpenAI-client boilerplate and drafting docstrings, but the key engineering decisions came from  iterating against real output : I ran the three ground-truth questions, saw the Client Credentials question wrongly
fall back, inspected the actual retrieved chunks, and used that evidence toredesign the prompt (the factual-vs-capability split). I treated every LLM suggestion as a hypothesis to verify — running `evaluate.py` and a headless Streamlit `AppTest` after each change rather than trusting the code on sight.

  ## Why I'm a strong fit for the ProAnalyst AI team

 I build production RAG, not demos. 
 This project has the things that matter in production — idempotent ingestion, cached models, typed error handling, measured latency, and a grounded prompt with a verifiable anti-hallucination guarantee — not just a working happy path.

I verify instead of assuming.  
I diagnosed the Q3 behaviour by reading the retrieved context, fixed the caching bug by tracing the call path, and proved every change with automated evaluation and UI smoke tests. That  evidence-driven habit is exactly what AI features need to be trustworthy.

I care about correctness and security. 
 I kept the API key out of source  entirely (`.env` + `.gitignore`), made the bot refuse rather than guess, and
  documented the trade-offs — the discipline ProAnalyst needs when shipping AI that real users and analysts depend on.


I believe I am a strong fit for the ProAnalyst AI team because of my combination of practical AI experience, research background, and eagerness to learn and contribute.

First, I have gained hands-on industry experience through multiple AI and automation internships. I worked as an AI Engineer Intern at LearnLine Edustation, where I contributed to AI pipelines, workflow automation, prompt engineering, and real-world AI systems. I also worked as an AI/ML Engineer Intern at GradeLab, where I was exposed to machine learning workflows and AI-driven product development. Additionally, as a Process Automation Intern at Champion InfoMetrics, I gained experience in automation projects and working within professional development environments. These experiences have strengthened my ability to apply AI concepts to real business problems.

Second, I have a strong academic and research foundation in Artificial Intelligence and Machine Learning. I am pursuing a B.Tech in Computer Science Engineering with a specialization in AI & ML at Dayananda Sagar University. I have also authored and published research papers in areas such as renewable energy forecasting, network intrusion detection, and deep learning applications. Through these projects, I have developed strong analytical thinking, problem-solving skills, and a deep understanding of machine learning techniques.

Finally, I am passionate about building practical AI solutions. I have worked on projects involving NLP, computer vision, deep learning, YOLO-based detection systems, chatbot development, and predictive analytics. I enjoy learning new technologies, collaborating with teams, and taking ownership of challenging tasks. I am confident that my technical skills, research mindset, and dedication to continuous improvement would allow me to contribute effectively to the ProAnalyst AI team while also growing as an AI professional.


