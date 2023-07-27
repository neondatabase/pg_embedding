ARG PG_MAJOR=15
FROM postgres:$PG_MAJOR
ARG PG_MAJOR

COPY . /tmp/pg_embedding

# Locale settings
RUN apt-get clean && apt-get update && \
    apt-get install -y locales && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

RUN apt-get install -y --no-install-recommends build-essential postgresql-server-dev-$PG_MAJOR && \
        cd /tmp/pg_embedding && \
        make clean && \
        make OPTFLAGS="" && \
        make install && \
        mkdir /usr/share/doc/pg_embedding && \
        cp LICENSE README.md /usr/share/doc/pg_embedding && \
        rm -r /tmp/pg_embedding && \
        apt-get remove -y build-essential postgresql-server-dev-$PG_MAJOR && \
        apt-get autoremove -y && \
        rm -rf /var/lib/apt/lists/*
