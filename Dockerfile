FROM python:3.8

RUN mkdir -p /diamond
COPY ./diamond /diamond

RUN pip install pycaret
RUN python -m ipykernel install --user --name diamond --display-name "diamond"
RUN pip install fastapi
RUN pip install uvicorn[standard]
RUN pip install "uvicorn[standard]"
RUN pip install gradio

EXPOSE 8000

CMD ["uvicorn", "diamond.diamond:app", "--host", "0.0.0.0", "--port", "8000"]

