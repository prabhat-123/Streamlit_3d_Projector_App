FROM python:3.8

CMD mkdir /streamit_3d_projector_app

COPY . /streamit_3d_projector_app

WORKDIR /streamit_3d_projector_app

EXPOSE 8501


RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt


## tell the image what to do when it starts as a container
CMD ["streamlit", "run", "src/main_v3.py"]