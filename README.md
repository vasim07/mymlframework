# ML Framework

Docker should be installed.

Clone this git repository.

`git clone https://github.com/vasim07/mymlframework`

Run the following docker command in your console
`docker run -it -v F:/filepath/input:/home/mlframework/input --publish 5000:5000 --publish 5555:5555 mlframe /bin/bash`

Await till docker builds up the container

In F:/filepath/input input your dataset with filename as `train.csv` and target feature as `target`.

Inside container, run `./docker_run.sh`
