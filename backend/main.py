from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import os
import asyncio
import uuid
import threading
from datetime import datetime
import logging
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


fake_users_db = {
    "admin": {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "password": "123",
        "avatar": None,
    }
}

DEFAULT_ROLES = [
    {"id": "wukong", "name": "孙悟空", "description": "齐天大圣，桀骜不驯、嫉恶如仇，口头禅"俺老孙"", "avatar": "🐒"},
    {"id": "bajie", "name": "猪八戒", "description": "天蓬元帅，贪吃好色但憨厚可爱", "avatar": "🐷"},
    {"id": "shaseng", "name": "沙悟净", "description": "卷帘大将，忠厚老实，任劳任怨", "avatar": "🧔"},
    {"id": "tangseng", "name": "唐僧", "description": "金蝉子转世，慈悲为怀，有时迂腐", "avatar": "🧘"},
]


class Config:
    def _find_model_path():
        possible_paths = [
            os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B"),
            "./models/Qwen/Qwen2.5-3B-Instruct",
            "./models/Qwen/Qwen3-0.6B",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return possible_paths[0]

    MODEL_NAME = os.getenv("MODEL_PATH", _find_model_path())
    LORA_PATH = os.getenv("LORA_PATH", "./outputs/qwen_roleplay")
    USE_LORA = os.getenv("USE_LORA", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 2048))
    MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 6))

    SYSTEM_PROMPT = """
你是一个角色扮演助手，请根据用户的输入，给出符合角色特点的回复。
"""


logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class User(BaseModel):
    id: int
    username: str
    email: str
    avatar: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    user: Optional[User] = None
    access_token: Optional[str] = None
    token_type: Optional[str] = None


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: user, assistant")
    content: str = Field(..., description="消息内容")
    time: Optional[str] = Field(None, description="消息时间")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="对话历史")
    max_tokens: Optional[int] = Field(512, ge=1, le=2048, description="最大生成长度")
    temperature: Optional[float] = Field(0.7, ge=0.1, le=2.0, description="温度参数")
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1.0, description="Top-p采样参数")
    repetition_penalty: Optional[float] = Field(
        1.1, ge=1.0, le=2.0, description="重复惩罚"
    )


class ChatResponse(BaseModel):
    role: str = Field(..., description="回复角色")
    content: str = Field(..., description="回复内容")
    time: str = Field(..., description="回复时间")
    status: str = Field("success", description="响应状态")
    tokens_used: Optional[int] = Field(None, description="使用的token数量")


class SimpleChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    max_tokens: Optional[int] = Field(512, ge=1, le=2048, description="最大生成长度")


class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否加载")
    timestamp: str = Field(..., description="检查时间")
    device: Optional[str] = Field(None, description="模型运行设备")


class CrawlRequest(BaseModel):
    url: str = Field(..., description="要爬取的网站URL")
    output_dir: Optional[str] = Field(None, description="输出目录")


class FineTuneRequest(BaseModel):
    url: Optional[str] = Field(None, description="数据源URL")
    split_mode: str = Field("paragraph", description="RAG切片方式: file, fixed, paragraph")
    target_role: Optional[str] = Field(None, description="目标角色名称")
    chunk_size: Optional[int] = Field(500, description="固定切片大小")
    chunk_overlap: Optional[int] = Field(50, description="固定切片重叠")


class PipelineStepStatus(BaseModel):
    id: str
    name: str
    status: str = "idle"
    message: str = ""
    duration: Optional[float] = None


class PipelineStatusResponse(BaseModel):
    steps: List[PipelineStepStatus]


class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.is_loaded = False

   
        try:
            logger.info("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME, trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("正在加载基础模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            if Config.USE_LORA and os.path.exists(Config.LORA_PATH):
                logger.info(f"正在加载LoRA权重: {Config.LORA_PATH}")
                self.model = PeftModel.from_pretrained(
                    self.model, Config.LORA_PATH, torch_dtype=torch.float16
                )
                logger.info("LoRA权重加载成功")
            elif Config.USE_LORA:
                logger.warning(f"LoRA路径不存在: {Config.LORA_PATH}，仅使用基础模型")

            self.model.eval()
            self.device = self.model.device
            self.is_loaded = True

            logger.info(f"模型加载完成！设备: {self.device}")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            self.is_loaded = False
            raise e

    def generate_response(
        self, messages: List[dict], generation_config: dict
    ) -> tuple[str, int]:
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logger.debug(f"格式化后的提示: {formatted_prompt}")

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_INPUT_LENGTH,
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]
        logger.info(f"输入token数量: {input_tokens}")

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)

        response_tokens = outputs[0][input_tokens:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        response_text = response_text.strip()
        total_tokens = outputs[0].shape[0]

        return response_text, total_tokens


model_manager = ModelManager()

pipeline_tasks: Dict[str, Dict] = {}


def _run_pipeline(task_id: str, config: FineTuneRequest):
    steps = [
        {"id": "crawl", "name": "网络爬取"},
        {"id": "extract", "name": "对话提取"},
        {"id": "filter", "name": "质量筛选"},
        {"id": "reconstruct", "name": "数据重构"},
        {"id": "train", "name": "模型微调"},
        {"id": "rag", "name": "RAG知识库构建"},
    ]
    task = pipeline_tasks[task_id]
    task["steps"] = {s["id"]: {"name": s["name"], "status": "idle", "message": "", "duration": None} for s in steps}

    def update_step(step_id, status, message="", duration=None):
        task["steps"][step_id]["status"] = status
        task["steps"][step_id]["message"] = message
        task["steps"][step_id]["duration"] = duration

    try:
        if config.url:
            update_step("crawl", "running", "正在爬取网站...")
            import subprocess
            result = subprocess.run(
                ["python", "crawler.py"],
                capture_output=True, text=True, timeout=600,
                env={**os.environ, "CRAWL_URL": config.url},
            )
            if result.returncode == 0:
                update_step("crawl", "success", "爬取完成")
            else:
                update_step("crawl", "error", f"爬取失败: {result.stderr[:200]}")
                task["status"] = "error"
                return

        update_step("extract", "running", "正在提取对话...")
        from Extract import Extract
        extract = Extract()
        asyncio.run(extract.run())
        update_step("extract", "success", "提取完成")

        update_step("filter", "running", "正在筛选语料...")
        from Filter import Filter
        role_name = config.target_role or "孙悟空"
        filter_inst = Filter(role_name)
        asyncio.run(filter_inst.run())
        update_step("filter", "success", "筛选完成")

        update_step("reconstruct", "running", "正在重构数据...")
        from Reconstruct import Reconstructor
        reconstructor = Reconstructor()
        asyncio.run(reconstructor.run())
        update_step("reconstruct", "success", "重构完成")

        update_step("train", "running", "正在微调模型...")
        import subprocess
        result = subprocess.run(
            ["python", "train_model.py"],
            capture_output=True, text=True, timeout=7200,
        )
        if result.returncode == 0:
            update_step("train", "success", "微调完成")
        else:
            update_step("train", "error", f"微调失败: {result.stderr[:200]}")
            task["status"] = "error"
            return

        update_step("rag", "running", "正在构建RAG知识库...")
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from RAG.embedding import RagConstruction
        rag = RagConstruction(split_mode=config.split_mode, batch_size=16)
        rag.data = rag.load_data()
        rag.save_embeddings()
        update_step("rag", "success", "RAG知识库构建完成")

        task["status"] = "completed"
    except Exception as e:
        logger.error(f"流水线执行失败: {e}", exc_info=True)
        for sid, sinfo in task["steps"].items():
            if sinfo["status"] == "running":
                sinfo["status"] = "error"
                sinfo["message"] = str(e)
        task["status"] = "error"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_manager.load_model()
        logger.info("应用启动完成")
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        model_manager.is_loaded = False

    yield

    if model_manager.model is not None:
        del model_manager.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("模型资源已释放")


app = FastAPI(
    title="角色扮演AI聊天API",
    description="基于Qwen3和LoRA微调的对话API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="根路径")
async def root():
    return {
        "message": "角色扮演AI聊天API服务运行中",
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
        "version": "1.0.0",
    }


@app.post("/api/auth/login", response_model=LoginResponse, summary="登录")
async def login(login_request: LoginRequest):
    try:
        logging.info(f"正在登录用户: {login_request.username}")
        user = authenticate_user(login_request.username, login_request.password)
        if not user:
            return LoginResponse(success=False, message="用户名或密码错误")

        user_info = User(
            id=user["id"],
            username=user["username"],
            email=user["email"],
            avatar=user["avatar"],
        )
        return LoginResponse(
            success=True,
            user=user_info,
        )
    except Exception as e:
        logging.error(f"登录失败: {str(e)}")
        return LoginResponse(success=False, message="登录过程中发生错误")


@app.get("/api/roles", summary="获取角色列表")
async def get_roles():
    return {"roles": DEFAULT_ROLES}


@app.get("/api/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        model_loaded=model_manager.is_loaded,
        timestamp=datetime.now().isoformat(),
        device=str(model_manager.device) if model_manager.is_loaded else None,
    )


@app.post("/api/chat", response_model=ChatResponse, summary="对话接口")
async def chat_completion(request: ChatRequest):
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型未加载完成，请稍后重试",
        )

    try:
        recent_messages = request.messages[-Config.MAX_HISTORY_MESSAGES :]
        if len(request.messages) > Config.MAX_HISTORY_MESSAGES:
            logger.info(
                f"对话历史截断: {len(request.messages)} -> {len(recent_messages)}"
            )

        has_system = any(msg.role == "system" for msg in recent_messages)
        if not has_system:
            system_message = ChatMessage(role="system", content=Config.SYSTEM_PROMPT)
            recent_messages = [system_message] + recent_messages

        messages_dict = [
            {"role": msg.role, "content": msg.content} for msg in recent_messages
        ]

        logger.info(f"处理消息数量: {len(messages_dict)}")

        generation_config = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": True,
            "pad_token_id": model_manager.tokenizer.pad_token_id,
            "eos_token_id": model_manager.tokenizer.eos_token_id,
            "repetition_penalty": request.repetition_penalty,
        }

        response_text, total_tokens = model_manager.generate_response(
            messages_dict, generation_config
        )

        logger.info(f"生成回复长度: {len(response_text)}, 总token数: {total_tokens}")

        return ChatResponse(
            role="assistant",
            content=response_text,
            time=datetime.now().strftime("%H:%M"),
            tokens_used=total_tokens,
        )

    except Exception as e:
        logger.error(f"生成失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成失败: {str(e)}",
        )


@app.post("/api/chat/simple", summary="简化版对话接口")
async def chat_simple(request: SimpleChatRequest):
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="模型未加载完成"
        )

    try:
        messages = [
            {"role": "system", "content": Config.SYSTEM_PROMPT},
            {"role": "user", "content": request.message},
        ]

        generation_config = {
            "max_new_tokens": request.max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": model_manager.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }

        response_text, total_tokens = model_manager.generate_response(
            messages, generation_config
        )

        return {
            "response": response_text,
            "time": datetime.now().strftime("%H:%M"),
            "tokens_used": total_tokens,
        }

    except Exception as e:
        logger.error(f"简化对话失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/api/pipeline/crawl", summary="启动爬虫")
async def start_crawl(request: CrawlRequest):
    try:
        from crawler_agent import NovelDownloader
        output_dir = request.output_dir or "./data/crawled"
        downloader = NovelDownloader(request.url, output_dir)
        downloader.parse_and_save()
        return {"message": "爬取完成", "output_dir": output_dir}
    except Exception as e:
        logger.error(f"爬取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/start", summary="启动微调流水线")
async def start_pipeline(request: FineTuneRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    pipeline_tasks[task_id] = {
        "status": "running",
        "config": request.model_dump(),
        "steps": {},
        "created_at": datetime.now().isoformat(),
    }

    thread = threading.Thread(
        target=_run_pipeline, args=(task_id, request), daemon=True
    )
    thread.start()

    return {"task_id": task_id}


@app.get("/api/pipeline/status/{task_id}", summary="查询流水线状态")
async def get_pipeline_status(task_id: str):
    if task_id not in pipeline_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = pipeline_tasks[task_id]
    steps = []
    step_order = ["crawl", "extract", "filter", "reconstruct", "train", "rag"]
    for sid in step_order:
        if sid in task["steps"]:
            s = task["steps"][sid]
            steps.append(PipelineStepStatus(
                id=sid, name=s["name"], status=s["status"],
                message=s["message"], duration=s.get("duration"),
            ))

    return PipelineStatusResponse(steps=steps)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return user_dict


def authenticate_user(username: str, password: str, db=fake_users_db):
    user = get_user(db, username)
    if not user:
        return None
    if user["password"] != password:
        return None
    return user


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
        reload=False,
    )
