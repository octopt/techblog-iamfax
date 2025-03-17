from strawberry.django.views import GraphQLView

from django.urls import path
from django.views.decorators.csrf import csrf_exempt

# from webapps import schema
from .schema import schema

urlpatterns = [
    path(
        "graphql/",
        csrf_exempt(GraphQLView.as_view(schema=schema, graphql_ide="apollo-sandbox")),
    ),
]
