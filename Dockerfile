FROM python:3.13-slim

ARG MESA_VERSION=3.3.1

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN pip install --no-cache-dir \
    "mesa==${MESA_VERSION}" \
    jupyter \
    matplotlib \
    seaborn \
    networkx \
    altair \
    solara \
    "ipyvuetify<3"

# Mesa 3.3.1 bug in mpl_space_drawing.py: "z_order == zorder & mask_marker" parses as
# "z_order == (zorder & mask_marker)", so agents with a zorder are not drawn.
# Same bug in wolf and sheep problem.
RUN f=/usr/local/lib/python3.13/site-packages/mesa/visualization/mpl_space_drawing.py \
    && grep -q "zorder_mask = z_order == zorder & mask_marker" $f \
    && sed -i "s/zorder_mask = z_order == zorder & mask_marker/zorder_mask = (z_order == zorder) \& mask_marker/" $f

WORKDIR /app
