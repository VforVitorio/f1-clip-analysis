IMAGE := f1-clip-analysis

build:
	docker build -t $(IMAGE) .

run-preclip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python src/preclip/run_preclip.py

run-clip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		-w /opt/project \
		$(IMAGE) python src/clip/run_clip.py

compare:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		$(IMAGE) python scripts/compare_results.py

shell:
	docker run -it --rm --gpus all \
		-v "$(PWD):/opt/project" \
		$(IMAGE) /bin/bash

clean:
	docker rmi $(IMAGE)