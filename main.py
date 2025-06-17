# app.py - Resume Podcast AI (단일 파일 버전)
import asyncio  # ✨ asyncio.to_thread를 사용하기 위해 추가
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import tempfile
import os
import logging
import uuid
from pathlib import Path
from datetime import datetime
import re

# Podcastfy imports (기존 방식 그대로)
try:
    from podcastfy.client import generate_podcast
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    print(f"⚠️  필수 라이브러리 설치 필요: pip install podcastfy python-dotenv")
    print(f"Import error: {e}")

# PDF/DOCX 파싱용
try:
    import PyPDF2
    import docx
except ImportError:
    print("⚠️  파일 파싱 라이브러리 설치 필요: pip install PyPDF2 python-docx")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Pydantic 모델들 ====================
class ResumeAnalysis(BaseModel):
    resume_text: str = Field(..., description="추출된 이력서 텍스트")
    word_count: int = Field(..., description="단어 수")
    detected_skills: List[str] = Field(default=[], description="감지된 기술 스택")
    experience_level: str = Field(..., description="경력 수준")
    suggested_interview_type: str = Field(..., description="추천 면접 유형")
    key_projects: List[str] = Field(default=[], description="주요 프로젝트")
    analysis_summary: str = Field(..., description="분석 요약")

# ==================== 기존 Gradio 방식 함수들 ====================
def get_api_key(key_name, ui_value):
    """기존 Gradio의 get_api_key 함수 그대로"""
    return ui_value if ui_value else os.getenv(key_name)

def create_conversation_config(
    word_count=2000,
    conversation_style="professional,encouraging",
    roles_person1="경험 많은 면접관",
    roles_person2="열정적인 지원자",
    dialogue_structure="자기소개,경험탐구,마무리",
    podcast_name="INTERVIEW PODCAST",
    podcast_tagline="면접 시뮬레이션",
    creativity_level=0.7,
    user_instructions=""
):
    """기존 Gradio의 conversation_config 구조 그대로"""
    return {
        "word_count": word_count,
        "conversation_style": conversation_style.split(',') if isinstance(conversation_style, str) else conversation_style,
        "roles_person1": roles_person1,
        "roles_person2": roles_person2,
        "dialogue_structure": dialogue_structure.split(',') if isinstance(dialogue_structure, str) else dialogue_structure,
        "podcast_name": podcast_name,
        "podcast_tagline": podcast_tagline,
        "creativity": creativity_level,
        "user_instructions": user_instructions
    }

# 면접 유형별 프리셋
INTERVIEW_PRESETS = {
    "tech": {
        "roles_person1": "경력 10년의 따뜻하고 전문적인 시니어 개발자 면접관",
        "roles_person2": "성장 의지가 강하고 기술에 열정적인 개발자 지원자",
        "conversation_style": "professional,encouraging,technical",
        "dialogue_structure": "따뜻한 인사와 자기소개,기술 스택과 개발 경험 탐구,주요 프로젝트 상세 분석,미래 목표와 비전,격려와 긍정적 마무리",
        "podcast_name": "TECH INTERVIEW SIMULATION",
        "podcast_tagline": "당신의 기술적 여정을 들려주세요"
    },
    "design": {
        "roles_person1": "경험 많은 크리에이티브 디렉터 면접관",
        "roles_person2": "창의적이고 열정적인 디자이너 지원자",
        "conversation_style": "creative,inspiring,professional",
        "dialogue_structure": "자기소개와 디자인 철학,포트폴리오 프로젝트 분석,창작 과정과 영감,디자인 트렌드와 미래 비전,격려와 마무리",
        "podcast_name": "DESIGN INTERVIEW SIMULATION",
        "podcast_tagline": "당신의 창작 세계를 보여주세요"
    },
    "business": {
        "roles_person1": "경험 많은 비즈니스 리더 면접관",
        "roles_person2": "전략적 사고력을 갖춘 비즈니스 지원자",
        "conversation_style": "strategic,analytical,professional",
        "dialogue_structure": "자기소개와 비즈니스 경험,전략적 사고와 문제해결,리더십과 팀워크 경험,시장 이해와 미래 계획,격려와 마무리",
        "podcast_name": "BUSINESS INTERVIEW SIMULATION",
        "podcast_tagline": "당신의 비즈니스 역량을 펼쳐보세요"
    }
}

def generate_interview_podcast_original_way(
    text_input: str,
    gemini_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    elevenlabs_key: Optional[str] = None,
    tts_model: str = "openai",
    word_count: int = 2000,
    creativity_level: float = 0.7,
    interview_type: str = "tech",
    difficulty_level: str = "mid",
    **kwargs
) -> str:
    """기존 Gradio 방식 그대로 팟캐스트 생성"""
    
    logger.info("Starting podcast generation process (Original Gradio Way)")
    
    # API key handling (기존 방식)
    logger.debug("Setting API keys")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    
    if tts_model == "openai" and openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    elif tts_model == "elevenlabs" and elevenlabs_key:
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
    
    # 면접 유형별 설정
    preset = INTERVIEW_PRESETS.get(interview_type, INTERVIEW_PRESETS["tech"])
    
    # 난이도별 설정
    difficulty_instructions = {
        "junior": "신입/주니어 수준에 맞춰 기본적이고 격려적인 분위기로 진행해주세요.",
        "mid": "중급 수준에 맞춰 실무 경험과 성장 가능성에 집중해주세요.",
        "senior": "시니어 수준에 맞춰 심화된 질문과 리더십에 대해 다뤄주세요."
    }
    
    user_instructions = difficulty_instructions.get(difficulty_level, difficulty_instructions["mid"])
    
    # Prepare conversation config (기존 방식)
    conversation_config = create_conversation_config(
        word_count=word_count,
        conversation_style=preset["conversation_style"],
        roles_person1=preset["roles_person1"],
        roles_person2=preset["roles_person2"],
        dialogue_structure=preset["dialogue_structure"],
        podcast_name=preset["podcast_name"],
        podcast_tagline=preset["podcast_tagline"],
        creativity_level=creativity_level,
        user_instructions=user_instructions
    )
    
    # Generate podcast (기존 방식 그대로)
    logger.info("Calling generate_podcast function")
    
    audio_file = generate_podcast(
        urls=None,  # URL 없음
        text=text_input,  # 이력서 텍스트
        image_paths=None,  # 이미지 없음
        tts_model=tts_model,
        conversation_config=conversation_config
    )
    
    logger.info("Podcast generation completed")
    return audio_file

# ==================== 이력서 파싱 함수들 ====================
def parse_resume_text(file_path: str) -> str:
    """이력서 파일에서 텍스트 추출"""
    file_path = Path(file_path)
    
    try:
        if file_path.suffix.lower() == '.pdf':
            return parse_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return parse_docx(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")
    except Exception as e:
        logger.error(f"파일 파싱 오류: {str(e)}")
        return "파일을 읽을 수 없습니다. 텍스트로 직접 입력해주세요."

def parse_pdf(file_path: Path) -> str:
    """PDF 파일에서 텍스트 추출"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return clean_text(text)
    except Exception as e:
        logger.error(f"PDF 파싱 오류: {str(e)}")
        return "PDF 파일을 읽을 수 없습니다."

def parse_docx(file_path: Path) -> str:
    """DOCX 파일에서 텍스트 추출"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return clean_text(text)
    except Exception as e:
        logger.error(f"DOCX 파싱 오류: {str(e)}")
        return "Word 문서를 읽을 수 없습니다."

def clean_text(text: str) -> str:
    """텍스트 정리"""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def analyze_resume_content(resume_text: str) -> Dict[str, Any]:
    """이력서 내용 간단 분석"""
    text_lower = resume_text.lower()
    
    # 기술 스택 추출
    tech_keywords = ['python', 'javascript', 'react', 'java', 'node.js', 'aws', 'docker', 'git']
    found_skills = [skill for skill in tech_keywords if skill in text_lower]
    
    # 경력 수준 판단
    if any(word in text_lower for word in ['시니어', 'senior', '팀장', '리드']):
        experience_level = "시니어"
    elif any(word in text_lower for word in ['신입', 'junior', '졸업']):
        experience_level = "신입"
    else:
        experience_level = "중급"
    
    # 면접 유형 추천
    if any(word in text_lower for word in ['개발', 'developer', '프로그래밍']):
        interview_type = "기술면접"
    elif any(word in text_lower for word in ['디자인', 'design', 'ui', 'ux']):
        interview_type = "디자인면접"
    else:
        interview_type = "비즈니스면접"
    
    return {
        "skills": found_skills[:5],
        "experience_level": experience_level,
        "interview_type": interview_type,
        "projects": ["프로젝트 경험"],  # 기본값
        "summary": f"{experience_level} 수준의 {interview_type} 이력서입니다."
    }

# ==================== FastAPI 앱 설정 ====================
app = FastAPI(
    title="Resume Podcast AI",
    description="AI-powered interview simulation podcast generator",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 생성
for dir_name in ["static", "static/audio", "temp"]:
    Path(dir_name).mkdir(exist_ok=True)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 진행 중인 작업 추적
active_tasks: Dict[str, Dict[str, Any]] = {}

# ==================== API 엔드포인트들 ====================
@app.get("/")
async def root():
    return {
        "message": "🎙️ Resume Podcast AI Backend is running!",
        "version": "1.0.0",
        "endpoints": {
            "analyze_resume": "POST /api/analyze-resume",
            "generate_podcast": "POST /api/generate-podcast", 
            "task_status": "GET /api/task-status/{task_id}",
            "download_audio": "GET /api/download-audio/{task_id}",
            "docs": "GET /docs"
        }
    }

@app.post("/api/analyze-resume", response_model=ResumeAnalysis)
async def analyze_resume(file: UploadFile = File(...)):
    """이력서 분석"""
    try:
        logger.info(f"이력서 분석 시작: {file.filename}")
        
        # 파일 확장자 체크
        if not file.filename.lower().endswith(('.pdf', '.docx', '.doc')):
            raise HTTPException(status_code=400, detail="PDF 또는 Word 문서만 지원됩니다.")
        
        # 파일 저장
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # 텍스트 추출
            resume_text = parse_resume_text(temp_path)
            
            if len(resume_text.strip()) < 20:
                raise HTTPException(status_code=400, detail="이력서에서 충분한 텍스트를 추출할 수 없습니다.")
            
            # 분석
            analysis = analyze_resume_content(resume_text)
            
            return ResumeAnalysis(
                resume_text=resume_text[:1500] + "..." if len(resume_text) > 1500 else resume_text,
                word_count=len(resume_text.split()),
                detected_skills=analysis.get("skills", []),
                experience_level=analysis.get("experience_level", "중급"),
                suggested_interview_type=analysis.get("interview_type", "기술면접"),
                key_projects=analysis.get("projects", []),
                analysis_summary=analysis.get("summary", "분석 완료")
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"이력서 분석 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/generate-podcast")
async def generate_podcast_endpoint(
    background_tasks: BackgroundTasks,
    resume_text: str = Form(...),
    interview_type: str = Form(default="tech"),
    difficulty_level: str = Form(default="mid"),
    tts_model: str = Form(default="openai"),
    word_count: int = Form(default=2000),
    creativity_level: float = Form(default=0.7),
    gemini_api_key: Optional[str] = Form(default=None),
    openai_api_key: Optional[str] = Form(default=None),
    elevenlabs_api_key: Optional[str] = Form(default=None),
):
    """팟캐스트 생성 시작"""
    try:
        task_id = str(uuid.uuid4())
        
        active_tasks[task_id] = {
            "status": "started",
            "progress": 0,
            "message": "팟캐스트 생성을 시작합니다...",
            "created_at": datetime.now().isoformat(),
            "audio_file": None,
            "error": None
        }
        
        background_tasks.add_task(
            generate_podcast_background,
            task_id, resume_text, interview_type, difficulty_level,
            tts_model, word_count, creativity_level,
            gemini_api_key, openai_api_key, elevenlabs_api_key
        )
        
        return {"task_id": task_id, "message": "팟캐스트 생성이 시작되었습니다."}
        
    except Exception as e:
        logger.error(f"팟캐스트 생성 시작 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_podcast_background(
    task_id: str, resume_text: str, interview_type: str, difficulty_level: str,
    tts_model: str, word_count: int, creativity_level: float,
    gemini_key: Optional[str], openai_key: Optional[str], elevenlabs_key: Optional[str]
):
    """백그라운드 팟캐스트 생성"""
    try:
        # 진행률 업데이트
        active_tasks[task_id].update({
            "status": "processing",
            "progress": 20,
            "message": "이력서 내용을 분석하고 있습니다..."
        })
        
        # 잠시 대기 (시뮬레이션)
        import asyncio
        await asyncio.sleep(1)
        
        active_tasks[task_id].update({
            "progress": 50,
            "message": "AI가 면접 대화를 생성하고 있습니다..."
        })
        
        # ✨ --- 핵심 수정 사항 --- ✨
        # 시간이 오래 걸리는 동기 함수를 별도의 스레드에서 실행하여
        # 메인 이벤트 루프를 막지 않고, asyncio.run() 충돌을 방지합니다.
        audio_file = await asyncio.to_thread(
            generate_interview_podcast_original_way,
            text_input=resume_text,
            interview_type=interview_type,
            difficulty_level=difficulty_level,
            tts_model=tts_model,
            word_count=word_count,
            creativity_level=creativity_level,
            gemini_key=gemini_key,
            openai_key=openai_key,
            elevenlabs_key=elevenlabs_key
        )
        
        active_tasks[task_id].update({
            "progress": 90,
            "message": "오디오 파일을 처리하고 있습니다..."
        })
        
        # 생성된 오디오 파일을 static 디렉토리로 이동
        if audio_file and os.path.exists(audio_file):
            file_extension = Path(audio_file).suffix
            new_filename = f"interview_{task_id}{file_extension}"
            new_path = Path("static/audio") / new_filename
            
            # 파일 이동
            os.rename(audio_file, new_path)
            
            # 작업 완료
            active_tasks[task_id].update({
                "status": "completed",
                "progress": 100,
                "message": "🎉 팟캐스트 생성이 완료되었습니다!",
                "audio_file": str(new_path),
                "download_url": f"/api/download-audio/{task_id}",
                "completed_at": datetime.now().isoformat()
            })
        else:
            raise Exception("오디오 파일 생성에 실패했습니다.")
            
    except Exception as e:
        logger.error(f"팟캐스트 생성 오류 (Task {task_id}): {str(e)}")
        active_tasks[task_id].update({
            "status": "failed",
            "progress": 0,
            "message": f"❌ 오류가 발생했습니다: {str(e)}",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """작업 진행 상태 확인"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return active_tasks[task_id]

@app.get("/api/download-audio/{task_id}")
async def download_audio(task_id: str):
    """생성된 오디오 파일 다운로드"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    
    task = active_tasks[task_id]
    
    if task["status"] != "completed" or not task["audio_file"]:
        raise HTTPException(status_code=400, detail="오디오 파일이 아직 준비되지 않았습니다.")
    
    audio_path = Path(task["audio_file"])
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="오디오 파일을 찾을 수 없습니다.")
    
    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=f"interview_podcast_{task_id}.mp3"
    )

@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(active_tasks),
        "available_interview_types": list(INTERVIEW_PRESETS.keys())
    }

# ==================== 메인 실행 ====================
if __name__ == "__main__":
    import uvicorn
    print("🎙️ Resume Podcast AI 시작...")
    print("📍 서버 주소: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("🛑 서버 종료: Ctrl + C")
    
    uvicorn.run(
        app,  # 문자열 대신 직접 앱 객체 전달
        host="0.0.0.0",
        port=8000,
        reload=False,  # 직접 실행 시에는 reload 비활성화
        log_level="info"
    )
