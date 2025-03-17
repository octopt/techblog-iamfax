import strawberry

from .graphql.mutations.mutations import (
    Mutation,
)  # フォルダ構成はまかせるが、上の説明と矛盾がないように
from .graphql.queries.queries import Query

schema = strawberry.Schema(query=Query, mutation=Mutation)
