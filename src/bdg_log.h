#ifndef BDG_LOG_H
#define BDG_LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/**
 * @file bdg_log.h
 * @brief Internal colored logging macros. All output to stderr.
 *
 * BDG_ERROR — red bold, prints [file:line], calls exit(EXIT_FAILURE)
 * BDG_WARN  — yellow bold, prints [file:line], continues
 * BDG_INFO  — cyan, user-facing (no file:line)
 */

static inline int bdg_use_color(void) {
    static int cached = -1;
    if (-1 == cached)
        cached = isatty(STDERR_FILENO);
    return cached;
}

#define BDG_RED_BOLD    "\033[1;31m"
#define BDG_YELLOW_BOLD "\033[1;33m"
#define BDG_CYAN        "\033[0;36m"
#define BDG_RESET       "\033[0m"

#define BDG_ERROR(fmt, ...) do { \
    if (bdg_use_color()) \
        fprintf(stderr, BDG_RED_BOLD "ERROR" BDG_RESET " [%s:%d] " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
    else \
        fprintf(stderr, "ERROR [%s:%d] " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
    exit(EXIT_FAILURE); \
} while (0)

#define BDG_WARN(fmt, ...) do { \
    if (bdg_use_color()) \
        fprintf(stderr, BDG_YELLOW_BOLD "WARN" BDG_RESET " [%s:%d] " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
    else \
        fprintf(stderr, "WARN [%s:%d] " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
} while (0)

#define BDG_INFO(fmt, ...) do { \
    if (bdg_use_color()) \
        fprintf(stderr, BDG_CYAN "INFO" BDG_RESET " " fmt "\n", ##__VA_ARGS__); \
    else \
        fprintf(stderr, "INFO " fmt "\n", ##__VA_ARGS__); \
} while (0)

#endif /* BDG_LOG_H */
