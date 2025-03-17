# Strawberry Django で GraphQL の Hello World

このリポジトリは、Django と Strawberry GraphQL を使って、GraphQL API の基本的な実装を学ぶためのサンプルです。  
GraphQL の Query と Mutation を使って、シンプルな「Hello World」を返す API を構築します。

## 特徴
- 🍓 **Strawberry GraphQL** を使ったシンプルな GraphQL API
- 🐍 **Python 3.11** と **Django 5.0** で構築
- 🐳 **Docker** 対応（開発環境を簡単に構築可能）
- 📚 **初学者向け** のシンプルな実装

## 動作環境
requirements.lock 参照ください

## セットアップ

### Docker を使う場合
```bash
# リポジトリをクローン
git clone https://github.com/yourname/iamfax-techblog
cd iamfax-techblog/tutorials/django-strawberry-graphql/part1-basic-setup

# Docker コンテナを起動
docker-compose up --build