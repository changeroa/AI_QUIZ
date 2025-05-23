#docker compose up --build -d
#docker에서 새로 pip하면 필요
#docker exec -it fastapi-app bash
#pip install ###
#pip freeze > requirements.txt
#exit
#docker cp fastapi-app:/app/requirements.txt ./requirements.txt
#도커 종료 - 파일 수정하면 다시 켜야함
#docker compose down 
#docker ps -a 실행중인 도커
#docker logs fastapi-app 로그, 에러확인
#http://localhost:8000/docs 실행하고 조금 기다려야 열림, 1분 넘게 걸림

from fastapi import FastAPI
from routers import quiz_router, vector_router, upload_router, admin_router

app = FastAPI()

app.include_router(quiz_router, prefix="/quiz")
app.include_router(vector_router, prefix="/vector")
app.include_router(upload_router, prefix="/upload")
app.include_router(admin_router, prefix="/admin")
