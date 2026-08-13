#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "ggml-backend-devs.h"

#define XPU_PLUGIN_NAME_SIZE 10
#define MAX_CONFIG_LINE_SIZE 256
#define XPU_CONFIG_PATH "/usr/share/kytensor/llm-whitelists.conf"

card_id_t* read_card_config(int* count) {
    FILE* fp;
    char line[MAX_CONFIG_LINE_SIZE];
    card_id_t* cards = NULL;
    int capacity = 32;
    int idx = 0;

    // 尝试打开配置文件
    fp = fopen(XPU_CONFIG_PATH, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open config file: %s\n", XPU_CONFIG_PATH);
        return NULL;
    }

    // 分配初始内存
    cards = (card_id_t *)malloc(capacity * sizeof(card_id_t));
    if (!cards) {
        fclose(fp);
        return NULL;
    }

    while (fgets(line, sizeof(line), fp)) {
        // 跳过注释行和空行
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }

        // 移除行尾的换行符
        char* newline = strchr(line, '\n');
        if (newline) *newline = '\0';

        char* comment = strchr(line, '#');
        if (comment) *comment = '\0';

        // 解析行: vendor_id device_id plugin_name [description]
        card_id_t card = {0, 0, ""};
        char description[128] = {0};

        int parsed = sscanf(line, "0x%x 0x%x %15s %127[^\n]",
                           &card.vendor_id, &card.device_id,
                           card.plugin_name, description);

        // 如果解析成功（至少3个字段）
        if (parsed >= 3) {
            // 检查是否需要扩容
            if (idx >= capacity) {
                capacity *= 2;
                card_id_t* temp = (card_id_t *)realloc(cards, capacity * sizeof(card_id_t));
                if (!temp) {
                    free(cards);
                    fclose(fp);
                    return NULL;
                }
                cards = temp;
            }

            cards[idx] = card;
            idx++;
        }
    }

    fclose(fp);
    *count = idx;
    return cards;
}

int find_devs_via_sysfs(char *plugin) {
    DIR *dir;
    struct dirent *entry;
    char vendor_path[256], device_path[256];
    unsigned int vendor_id, device_id;
    FILE *fp;


    // 从配置文件读取设备列表
    int card_count = 0;
    card_id_t* cards = read_card_config(&card_count);
    if (!cards || card_count == 0) {
        fprintf(stderr, "ERROR: No valid card configurations found in %s\n",
                XPU_CONFIG_PATH);
        return 0;
    }
    const card_id_t *p = cards;

    // 打开PCI设备目录
    dir = opendir("/sys/bus/pci/devices/");
    if (!dir) {
        free(cards);
        return 0;
    }

    while ((entry = readdir(dir)) != NULL) {
        // 跳过 "." 和 ".."
        if (entry->d_name[0] == '.') continue;

        // 构建 vendor 文件路径
        snprintf(vendor_path, sizeof(vendor_path),
                 "/sys/bus/pci/devices/%s/vendor", entry->d_name);
        // 构建 device 文件路径
        snprintf(device_path, sizeof(device_path),
                "/sys/bus/pci/devices/%s/device", entry->d_name);

        // 读取 vendor ID
        fp = fopen(vendor_path, "r");
        if (!fp) {
	    fprintf(stderr, "ERROR: open file %s failed\n", vendor_path);
            continue;
        }
        if (fscanf(fp, "%x", &vendor_id) != 1) {
            fclose(fp);
            continue;
        }
        fclose(fp);

	// 读取 device ID
        fp = fopen(device_path, "r");
        if (!fp) {
            fprintf(stderr, "ERROR: open file %s failed\n", device_path);
            continue;
        }
        if (fscanf(fp, "%x", &device_id) != 1) {
            fclose(fp);
            continue;
        }
        fclose(fp);

        // 与名单匹配
        p = cards;
        while (p && p->vendor_id != 0) {
            if (vendor_id == p->vendor_id && device_id == p->device_id) {
                memcpy(plugin, p->plugin_name, XPU_PLUGIN_NAME_SIZE);
                closedir(dir);
		free(cards);
                return 1;
            }
            p++;
        }
    }
    closedir(dir);
    free(cards);
    return 0;
}
