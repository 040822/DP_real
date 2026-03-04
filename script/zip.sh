#!/bin/bash
# scripts/zip.sh

# 要忽略的目录列表（以空格分隔）
IGNORE_DIRS="_outputs data outputs"


# 构建排除参数
EXCLUDE_ARGS=""
for dir in $IGNORE_DIRS; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS -x \"$dir/*\""
done

# 压缩当前目录为 current_dir.zip，排除指定目录
eval zip -r DP_real.zip . $EXCLUDE_ARGS
