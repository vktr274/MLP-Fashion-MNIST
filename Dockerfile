FROM python:3.7.12-slim

WORKDIR /mlp

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /mlp/src

CMD ["jupyter", "notebook", "--port", "8888", "--ip", "0.0.0.0", "--allow-root", "--no-browser"]