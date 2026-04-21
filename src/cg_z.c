#define CTYPE c64
#define RTYPE f64
#define CTYPE_IS_COMPLEX
#define PREFIX z
#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define FN(name) CONCAT(PREFIX, name)
#include "cg_impl.inc"
