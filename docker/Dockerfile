ARG PG_MAJOR_VER=15

FROM postgres:$PG_MAJOR_VER AS build

ARG PG_MAJOR_VER

RUN apt-get update \
    && apt-get install build-essential postgresql-server-dev-$PG_MAJOR_VER -y

ADD https://github.com/neondatabase/pg_embedding/archive/refs/heads/main.tar.gz main.tar.gz

RUN tar --strip-components=1 -xvf main.tar.gz \
    && make \
    && make install

FROM postgres:$PG_MAJOR_VER

COPY --from=build /usr/lib/postgresql /usr/lib/postgresql
COPY --from=build /usr/share/postgresql /usr/share/postgresql
COPY ./init-extension.sh /docker-entrypoint-initdb.d/init-extension.sh

RUN chmod +x /docker-entrypoint-initdb.d/init-extension.sh
