# The HTTP server as a container: CPU-only by design, so it runs anywhere a container does.
# A GPU host is better served by `pip install "vegaml[serve]"` against its own torch build.
FROM python:3.12-slim

# torch's CPU wheel keeps the image a fraction of the CUDA one's size
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . /src
RUN pip install --no-cache-dir "/src[serve]" && rm -rf /src

# The checkpoint downloads on first request; mount a volume here so restarts keep it.
VOLUME /root/.cache/huggingface
ENV VEGAML_MODEL=""
# One torch thread by default, because this image is the CPU path and that is where it was
# measured fastest for the shipped reader: 258 ms against 1268 ms at torch's own default of four,
# with identical answers. docs/cpu.md has the sweep, and a reader that prefers more threads takes
# VEGAML_THREADS.
ENV VEGAML_THREADS="1"
# A calibration fitted on your own labels, mounted in and applied to every answer. Without it this
# serves the checkpoint's published probabilities and no review band.
ENV VEGAML_CALIBRATION=""
EXPOSE 8000
CMD vegaml-serve --host 0.0.0.0 --port 8000     ${VEGAML_MODEL:+--model "$VEGAML_MODEL"}     ${VEGAML_THREADS:+--threads "$VEGAML_THREADS"}     ${VEGAML_CALIBRATION:+--calibration "$VEGAML_CALIBRATION"}
