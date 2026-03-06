#define CTYPE f64
#define RTYPE f64
#define CTYPE_IS_REAL
#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define PREFIX d
#define FN(name) CONCAT(PREFIX, name)

#include "precond_impl.inc"
