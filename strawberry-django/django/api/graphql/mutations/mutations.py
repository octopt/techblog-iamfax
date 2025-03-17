import strawberry


@strawberry.type
class Mutation:
    @strawberry.mutation
    def hello(self, info) -> str:
        return "Hello from imfaxblog!"
