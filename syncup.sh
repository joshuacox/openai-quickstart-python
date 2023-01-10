#!/bin/bash
set -eux
syncr () {
  rsync -ave ssh $1 coopadmin@10.11.5.11:/unreal/openai-quickstart-python/
}

syncr images
syncr images2
