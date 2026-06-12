FROM python:3.12-bookworm

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# curl/libgl1/systemctl per the base template; the cairo + pango + font
# libraries are required by cairosvg to rasterise the bracket posters.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl libgl1 systemctl \
      libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 \
      libffi-dev fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

WORKDIR /workspace

RUN pip install --upgrade pip --user

CMD ["tail", "-f", "/dev/null"]
