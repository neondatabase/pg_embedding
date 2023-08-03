ARG PG_VERSION=15

FROM debian:stable-slim AS build

ARG PG_VERSION

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install git build-essential postgresql-server-dev-$PG_VERSION -y

WORKDIR /workspace

RUN git clone https://github.com/neondatabase/pg_embedding.git \
    && cd pg_embedding \
    && make \
    && make install

FROM postgres:$PG_VERSION

COPY --from=build /usr/lib/postgresql/ /usr/lib/postgresql/
COPY --from=build /usr/share/postgresql/ /usr/share/postgresql/
COPY ./init-extension.sh /docker-entrypoint-initdb.d/init-extension.sh

RUN chmod +x /docker-entrypoint-initdb.d/init-extension.sh