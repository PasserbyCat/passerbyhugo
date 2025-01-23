#!/usr/bin/env bash

# 檔名: create_post.sh
# 用法: ./create_post.sh "文章標題" [faith|code|book|blog]
#
# 範例:
#   ./create_post.sh "聖經默想" faith
#   ./create_post.sh "Go 語言筆記" code
#   ./create_post.sh "某書心得" book
#   ./create_post.sh "我的日常更新" blog
#   ./create_post.sh "我忘了帶參數"     (也會歸類到 blog)

# 第 1 個參數：文章標題
TITLE="$1"
# 第 2 個參數：分類（可留空，預設歸為 blog）
CATEGORY="$2"

# 檢查參數：若沒有標題，直接跳錯結束
if [ -z "$TITLE" ]; then
  echo "用法：$0 \"文章標題\" [faith|code|book|blog]"
  echo "範例：$0 \"聖經默想\" faith"
  exit 1
fi

# 將文章標題轉為 slug（空白 → -、大寫 → 小寫）
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g')

# 根據傳入的分類，設定儲存路徑與 categories
# 若沒傳第 2 個參數，或輸入非預期值，一律歸類為 blog
case "$CATEGORY" in
  faith)
    POST_DIR="content/posts/faith/${SLUG}"
    CATEGORY_NAME="信仰分享"
    ;;
  code)
    POST_DIR="content/posts/code/${SLUG}"
    CATEGORY_NAME="程式紀錄"
    ;;
  book)
    POST_DIR="content/posts/book/${SLUG}"
    CATEGORY_NAME="閱讀心得"
    ;;
  blog | "")
    POST_DIR="content/posts/blog/${SLUG}"
    CATEGORY_NAME=""  # blog 就不放任何分類
    ;;
  *)
    echo "未識別的分類：$CATEGORY"
    echo "使用預設的『blog』分類。"
    POST_DIR="posts/blog/${SLUG}"
    CATEGORY_NAME=""
    ;;
esac

# 建立目錄（若已存在則不重複建立）
mkdir -p "$POST_DIR"

# 最終要建立的文章路徑
POST_FILE="${POST_DIR}/index.zh-tw.md"

# 若已存在同名檔案，則顯示錯誤並結束，避免覆蓋
if [ -f "$POST_FILE" ]; then
  echo "錯誤：檔案已存在 -> $POST_FILE"
  exit 1
fi

# 取得當前日期（用於 Front Matter）
DATE=$(date +"%Y-%m-%dT%H:%M:%S%z")

# 開始寫入 Front Matter
cat <<EOF > "$POST_FILE"
---
title: "$TITLE"
date: "$DATE"
draft: true
EOF

# 根據是否有分類名稱來決定要不要加上 categories
if [ -n "$CATEGORY_NAME" ]; then
  echo "categories: [\"$CATEGORY_NAME\"]" >> "$POST_FILE"
else
  # blog 類型或空白時，就給一個空陣列
  echo "categories: []" >> "$POST_FILE"
fi

# 補上其他欄位
cat <<EOF >> "$POST_FILE"
tags: []
---

此處開始撰寫文章內容 ...
EOF

# 提示建立成功
echo "已成功建立新文章：$POST_FILE"
