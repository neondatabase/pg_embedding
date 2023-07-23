EXTENSION = embedding
EXTVERSION = 0.1.0

MODULE_big = embedding
DATA = $(wildcard *--*.sql)
OBJS = embedding.o hnswalg.o distfunc.o

TESTS = $(wildcard test/sql/*.sql)
REGRESS = $(patsubst test/sql/%.sql,%,$(TESTS))
REGRESS_OPTS = --inputdir=test --load-extension=embedding

# For auto-vectorization:
# - GCC&clang needs -Ofast or -O3: https://gcc.gnu.org/projects/tree-ssa/vectorization.html
PG_CFLAGS += -Ofast
PG_CXXFLAGS += -std=c++11
PG_LDFLAGS += -lstdc++

all: $(EXTENSION)--$(EXTVERSION).sql

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

dist:
	mkdir -p dist
	git archive --format zip --prefix=$(EXTENSION)-$(EXTVERSION)/ --output dist/$(EXTENSION)-$(EXTVERSION).zip master
