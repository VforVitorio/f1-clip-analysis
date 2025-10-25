IMAGE := f1-clip-analysis

build:
	docker build -t $(IMAGE) .

run-preclip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		$(IMAGE) python scripts/run_preclip.py

run-clip:
	docker run --rm --gpus all \
		-v "$(PWD):/opt/project" \
		$(IMAGE) python scripts/run_clip.py

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