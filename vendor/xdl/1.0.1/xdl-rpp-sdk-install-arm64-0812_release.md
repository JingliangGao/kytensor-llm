# XDL RPP SDK 安装说明

本文档包含两个 DEB 包在ARM64系统上安装说明：

1. `xdl-rpp-dkms_1.0.0_all.deb`
2. `xdl-rpp-runtime_1.0.1_arm64.deb`
3. `xdl-rpp-runtime-dev_1.0.0_all.deb`

md5值：

c8b41692c1a8d3083a94264a856b89fb  xdl-rpp-dkms_1.0.0_all.deb

edd4c751277fbb384edb66439f943260 xdl-rpp-runtime_1.0.1_arm64.deb

618fcb9b960f41eb053ce63c2cfee82e  xdl-rpp-runtime-dev_1.0.0_all.deb


变更说明：

      1、llama.cpp支持单进程模式，路径下tools\server，具体改动请参考 intergration code compare-server 代码比较报告

      2、更新ggml-rpp，支持S3/S4(需搭配`xdl-rpp-runtime_1.0.1_amd64.deb`使用)


参考代码：

20666fa9413bb21f01b1b5dda677aced llama.cpp-b8966-rpp-1.0.1.zip


配置llamacpp说明：

请参考：RPP_INTEGRATION_GUIDE.md文档

请参考：intergration code compare-server 代码对比

## deb包说明

## 1) `xdl-rpp-dkms_1.0.0_all.deb`

- 包名：`xdl-rpp-dkms`
- 版本：`1.0.0`
- 架构：`all`
- 作用：提供 RPP DKMS 驱动源码、固件文件和 udev 规则，安装后按目标机内核编译并安装驱动模块。
- DEB 包文件大小：`172.56 KiB`（`0.17 MiB`）
- 全部产物总大小：`1036.77 KiB`（`1.01 MiB`） 

<details>
<summary>全部产物（路径、大小，以具备版本为准）</summary>

- `./etc/udev/rules.d/80-rpp.rules` | `0.42 KiB`（`0.0004 MiB`）
- `./lib/firmware/rpp/config_oem.bin` | `1.73 KiB`（`0.0017 MiB`）
- `./lib/firmware/rpp/firmware_oem.bin` | `104.13 KiB`（`0.1017 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/Makefile` | `2.74 KiB`（`0.0027 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/detect_platform.sh` | `1.88 KiB`（`0.0018 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/dkms.conf` | `0.37 KiB`（`0.0004 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/git_version.txt` | `0.01 KiB`（`0.0000 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/pci-dma-compat.h` | `3.66 KiB`（`0.0036 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_cdev.h` | `1.33 KiB`（`0.0013 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_cdev_ctrl.h` | `8.99 KiB`（`0.0088 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_com.h` | `21.83 KiB`（`0.0213 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_debug.h` | `1.38 KiB`（`0.0013 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_dev.h` | `34.87 KiB`（`0.0341 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_dev_list.h` | `1.84 KiB`（`0.0018 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_dfx.h` | `5.08 KiB`（`0.0050 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_dma.h` | `14.03 KiB`（`0.0137 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_hal_dma.h` | `11.46 KiB`（`0.0112 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_hal_pcie.h` | `2.07 KiB`（`0.0020 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_isr.h` | `0.78 KiB`（`0.0008 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_pages.h` | `3.85 KiB`（`0.0038 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_pm.h` | `1.07 KiB`（`0.0010 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_sdev.h` | `1.05 KiB`（`0.0010 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_sdev_ctrl.h` | `2.68 KiB`（`0.0026 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_types.h` | `7.72 KiB`（`0.0075 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_vdev.h` | `1.00 KiB`（`0.0010 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_vdev_ctrl.h` | `23.94 KiB`（`0.0234 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_ve_vmm.h` | `2.03 KiB`（`0.0020 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/rpp_xdata.h` | `5.40 KiB`（`0.0053 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/include/version.h` | `0.92 KiB`（`0.0009 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/platform.mk` | `2.56 KiB`（`0.0025 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_cdev.c` | `8.90 KiB`（`0.0087 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_cdev_ctrl.c` | `39.67 KiB`（`0.0387 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_dev.c` | `107.68 KiB`（`0.1052 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_dev_list.c` | `35.47 KiB`（`0.0346 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_dma.c` | `39.49 KiB`（`0.0386 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_hal_dma.c` | `92.28 KiB`（`0.0901 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_hal_pcie.c` | `16.02 KiB`（`0.0156 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_isr.c` | `23.71 KiB`（`0.0232 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_mod.c` | `71.73 KiB`（`0.0701 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_mpu.c` | `7.17 KiB`（`0.0070 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_pages.c` | `21.38 KiB`（`0.0209 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_pm.c` | `20.72 KiB`（`0.0202 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_sdev.c` | `19.14 KiB`（`0.0187 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_sdev_ctrl.c` | `9.10 KiB`（`0.0089 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_vdev.c` | `15.02 KiB`（`0.0147 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_vdev_ctrl.c` | `212.91 KiB`（`0.2079 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_ve_vmm.c` | `14.91 KiB`（`0.0146 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/rpp_xdata.c` | `10.57 KiB`（`0.0103 MiB`）
- `./usr/src/rpp-dkms-2.0.16.6/version.mk` | `0.06 KiB`（`0.0001 MiB`）
- `./lib/firmware/rpp/config.bin -> config_oem.bin` | `0.00 KiB`（`0.0000 MiB`）
- `./lib/firmware/rpp/firmware.bin -> firmware_oem.bin` | `0.00 KiB`（`0.0000 MiB`）

</details>

## 2) `xdl-rpp-runtime_1.0.1_arm64.deb`

- 包名：`xdl-rpp-runtime`
- 版本：`1.0.1`
- 架构：`arm64`
- 作用：提供 RPP Runtime 库、头文件、系统配置和运行时配置文件。
- DEB 包文件大小：`1209.59 KiB`（`1.18 MiB`）
- 全部产物总大小：`5372.40 KiB`（`5.25 MiB`）

<details>
<summary>全部产物（路径、大小）</summary>

- `./etc/ld.so.conf.d/rpp_Kylin.conf` | `0.02 KiB`（`0.0000 MiB`）
- `./usr/include/xpu/rpp/__clang_cuda_builtin_vars.h` | `6.92 KiB`（`0.0068 MiB`）
- `./usr/include/xpu/rpp/bfloat16.h` | `15.83 KiB`（`0.0155 MiB`）
- `./usr/include/xpu/rpp/common.h` | `3.29 KiB`（`0.0032 MiB`）
- `./usr/include/xpu/rpp/cuda2rpp.h` | `3.08 KiB`（`0.0030 MiB`）
- `./usr/include/xpu/rpp/drvapi_error_string.h` | `17.21 KiB`（`0.0168 MiB`）
- `./usr/include/xpu/rpp/fifo_mutex.h` | `1.75 KiB`（`0.0017 MiB`）
- `./usr/include/xpu/rpp/gdev_list.h` | `3.39 KiB`（`0.0033 MiB`）
- `./usr/include/xpu/rpp/half.hpp` | `142.81 KiB`（`0.1395 MiB`）
- `./usr/include/xpu/rpp/msg_def.h` | `1.45 KiB`（`0.0014 MiB`）
- `./usr/include/xpu/rpp/pti_activity.h` | `166.42 KiB`（`0.1625 MiB`）
- `./usr/include/xpu/rpp/pti_callbacks.h` | `19.70 KiB`（`0.0192 MiB`）
- `./usr/include/xpu/rpp/pti_events.h` | `45.55 KiB`（`0.0445 MiB`）
- `./usr/include/xpu/rpp/pti_metrics.h` | `27.67 KiB`（`0.0270 MiB`）
- `./usr/include/xpu/rpp/pti_result.h` | `5.54 KiB`（`0.0054 MiB`）
- `./usr/include/xpu/rpp/pti_version.h` | `1.03 KiB`（`0.0010 MiB`）
- `./usr/include/xpu/rpp/rppFatBinary.h` | `5.83 KiB`（`0.0057 MiB`）
- `./usr/include/xpu/rpp/rpp_block_segment.h` | `3.95 KiB`（`0.0039 MiB`）
- `./usr/include/xpu/rpp/rpp_com.h` | `25.67 KiB`（`0.0251 MiB`）
- `./usr/include/xpu/rpp/rpp_driver_api_id.h` | `9.30 KiB`（`0.0091 MiB`）
- `./usr/include/xpu/rpp/rpp_driver_api_meta.h` | `34.66 KiB`（`0.0338 MiB`）
- `./usr/include/xpu/rpp/rpp_drv_api.h` | `266.96 KiB`（`0.2607 MiB`）
- `./usr/include/xpu/rpp/rpp_elf.h` | `2.10 KiB`（`0.0021 MiB`）
- `./usr/include/xpu/rpp/rpp_f16.h` | `10.60 KiB`（`0.0104 MiB`）
- `./usr/include/xpu/rpp/rpp_graph_mgr.h` | `7.48 KiB`（`0.0073 MiB`）
- `./usr/include/xpu/rpp/rpp_half.h` | `0.35 KiB`（`0.0003 MiB`）
- `./usr/include/xpu/rpp/rpp_inline.cuh` | `19.25 KiB`（`0.0188 MiB`）
- `./usr/include/xpu/rpp/rpp_math.h` | `26.19 KiB`（`0.0256 MiB`）
- `./usr/include/xpu/rpp/rpp_perf.h` | `32.31 KiB`（`0.0316 MiB`）
- `./usr/include/xpu/rpp/rpp_pipe.h` | `25.85 KiB`（`0.0252 MiB`）
- `./usr/include/xpu/rpp/rpp_pti.h` | `2.31 KiB`（`0.0023 MiB`）
- `./usr/include/xpu/rpp/rpp_runtime.h` | `169.53 KiB`（`0.1656 MiB`）
- `./usr/include/xpu/rpp/rpp_smap.h` | `7.77 KiB`（`0.0076 MiB`）
- `./usr/include/xpu/rpp/rpp_smgr.h` | `36.94 KiB`（`0.0361 MiB`）
- `./usr/include/xpu/rpp/rpp_sock.h` | `0.70 KiB`（`0.0007 MiB`）
- `./usr/include/xpu/rpp/rpp_sram_io.h` | `19.55 KiB`（`0.0191 MiB`）
- `./usr/include/xpu/rpp/rpp_tensor.hpp` | `15.83 KiB`（`0.0155 MiB`）
- `./usr/include/xpu/rpp/rpp_typedef.h` | `6.18 KiB`（`0.0060 MiB`）
- `./usr/include/xpu/rpp/rpp_types.h` | `3.01 KiB`（`0.0029 MiB`）
- `./usr/include/xpu/rpp/vector_types.h` | `6.90 KiB`（`0.0067 MiB`）
- `./usr/include/xpu/rpp/version.h` | `0.43 KiB`（`0.0004 MiB`）
- `./usr/lib/xpu/rpp/librpp_perf.so.1.0.0` | `60.28 KiB`（`0.0589 MiB`）
- `./usr/lib/xpu/rpp/liburpp.so.2.0.21.2` | `4099.91 KiB`（`4.0038 MiB`）
- `./usr/lib/xpu/rpp/liburpp.so -> liburpp.so.2` | `0.00 KiB`（`0.0000 MiB`）

</details>

- ## 安装与卸载

**安装xdl-rpp-dkms包**

```bash
sudo dpkg -i xdl-rpp-dkms_1.0.0_all.deb
```

xdl-rpp-dkms包安装日志：

<details>
<summary>展开/收起日志</summary>

```
正在选中未选择的软件包 xdl-rpp-dkms。
(正在读取数据库 ... 系统当前共安装有 251335 个文件和目录。)
准备解压 xdl-rpp-dkms_1-1_all.deb  ...
正在解压 xdl-rpp-dkms (1-1) ...
正在设置 xdl-rpp-dkms (1-1) ...

Creating symlink /var/lib/dkms/rpp-dkms/2.0.16.12/source ->
                 /usr/src/rpp-dkms-2.0.16.12

DKMS: add completed.

Kernel preparation unnecessary for this kernel.  Skipping...

Building module:
cleaning build area...
make -j8 KERNELRELEASE=5.4.18-155-generic -C /lib/modules/5.4.18-155-generic/build M=/var/lib/dkms/rpp-dkms/2.0.16.12/build variant=vpu modules...
cleaning build area...

DKMS: build completed.

rpp.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/5.4.18-155-generic/kernel/extra/

depmod...

DKMS: install completed.
```

</details>

xdl-rpp-dkms检查命令：

1、查询DKMS 驱动源码

2、查询DKMS 构建产物

```bash
ls -ll /usr/src/rpp-dkms-*/
ls -ll /lib/modules/$(uname -r)/kernel/extra/
```

检查命令日志：

<details>
<summary>展开/收起日志</summary>

```
gytest@gytest-pc:~/Yan_debug/0728_deb$ ls -ll /usr/src/rpp-dkms-*/
总用量 820
-rwxr-xr-x 1 root root   1923 7月  28 16:21 detect_platform.sh
-rw-r--r-- 1 root root    377 7月  28 16:21 dkms.conf
-rw-r--r-- 1 root root      8 7月  28 16:21 git_version.txt
drwxrwxr-x 2 root root   4096 7月  28 18:19 include
-rw-r--r-- 1 root root   2801 7月  28 16:21 Makefile
-rw-r--r-- 1 root root   2966 7月  28 16:21 platform.mk
-rw-r--r-- 1 root root   9114 7月  28 16:21 rpp_cdev.c
-rw-r--r-- 1 root root  39438 7月  28 16:21 rpp_cdev_ctrl.c
-rw-r--r-- 1 root root 113283 7月  28 16:21 rpp_dev.c
-rw-r--r-- 1 root root  36322 7月  28 16:21 rpp_dev_list.c
-rw-r--r-- 1 root root  40439 7月  28 16:21 rpp_dma.c
-rw-r--r-- 1 root root  94495 7月  28 16:21 rpp_hal_dma.c
-rw-r--r-- 1 root root  17186 7月  28 16:21 rpp_hal_pcie.c
-rw-r--r-- 1 root root  24283 7月  28 16:21 rpp_isr.c
-rw-r--r-- 1 root root  73645 7月  28 16:21 rpp_mod.c
-rw-r--r-- 1 root root   7345 7月  28 16:21 rpp_mpu.c
-rw-r--r-- 1 root root  21769 7月  28 16:21 rpp_pages.c
-rw-r--r-- 1 root root  16327 7月  28 16:21 rpp_pm.c
-rw-r--r-- 1 root root  19598 7月  28 16:21 rpp_sdev.c
-rw-r--r-- 1 root root   9319 7月  28 16:21 rpp_sdev_ctrl.c
-rw-r--r-- 1 root root  15384 7月  28 16:21 rpp_vdev.c
-rw-r--r-- 1 root root 218471 7月  28 16:21 rpp_vdev_ctrl.c
-rw-r--r-- 1 root root  15270 7月  28 16:21 rpp_ve_vmm.c
-rw-r--r-- 1 root root  10828 7月  28 16:21 rpp_xdata.c
-rw-r--r-- 1 root root     66 7月  28 16:21 version.mk
gytest@gytest-pc:~/Yan_debug/0728_deb$ ls -ll /lib/modules/$(uname -r)/kernel/extra/
总用量 468
-rw-r--r-- 1 root root 478896 7月  28 18:20 rpp.ko
gytest@gytest-pc:~/Yan_debug/0728_deb$ 
```

</details>

安装产物检查-firmware路径确认：

```
gytest@gytest-pc:~/Yan_debug/0729_release$   ls -ll /lib/firmware/rpp/
总用量 112
lrwxrwxrwx 1 root root     14 7月  29 15:33 config.bin -> config_oem.bin
-rwxrw-r-- 1 root root   1768 6月  17 10:26 config_oem.bin
lrwxrwxrwx 1 root root     16 7月  29 15:33 firmware.bin -> firmware_oem.bin
-rwxrw-r-- 1 root root 106628 7月  29 15:33 firmware_oem.bin
gytest@gytest-pc:~/Yan_debug/0729_release$ 
```


**安装xdl-rpp-runtime包:**

```bash
sudo dpkg -i xdl-rpp-runtime_1.0.1_arm64.deb
```

xdl-rpp-dkms包安装日志：

<details>
<summary>展开/收起日志</summary>

```
正在选中未选择的软件包 xdl-rpp-runtime。
(正在读取数据库 ... 系统当前共安装有 251389 个文件和目录。)
准备解压 xdl-rpp-runtime_1.0.1_arm64.deb  ...
正在解压 xdl-rpp-runtime (1.0.1) ...
正在设置 xdl-rpp-runtime (1.0.1) ...
```

</details>

安装产物检查-环境变量查询：

```
gytest@gytest-pc:~/Yan_debug/0729_release$ ls -ll /etc/ld.so.conf.d/rpp_Kylin.conf 
-rw-r--r-- 1 root root 17 7月  27 15:53 /etc/ld.so.conf.d/rpp_Kylin.conf
gytest@gytest-pc:~/Yan_debug/0729_release$ 
```

安装产物检查-头文件查询：

```
gytest@gytest-pc:~/Yan_debug/0729_release$ ls -ll /usr/include/xpu/rpp/ | head
总用量 1268
-rw-r--r-- 1 root root  16213 7月  16  2025 bfloat16.h
-rw-r--r-- 1 root root   7086 11月  6  2024 __clang_cuda_builtin_vars.h
-rw-r--r-- 1 root root   3372 6月   6 23:31 common.h
-rw-r--r-- 1 root root   3150 11月  6  2024 cuda2rpp.h
-rw-r--r-- 1 root root  17620 4月  30 15:16 drvapi_error_string.h
-rw-r--r-- 1 root root   1794 7月  11  2025 fifo_mutex.h
-rw-r--r-- 1 root root   3467 7月  11  2025 gdev_list.h
-rw-r--r-- 1 root root 146237 11月  6  2024 half.hpp
-rw-r--r-- 1 root root   1489 7月  11  2025 msg_def.h
gytest@gytest-pc:~/Yan_debug/0729_release$ 
```

安装产物检查-运行时库查询：

```
gytest@gytest-pc:~/Yan_debug/0729_release$  ls -ll /usr/lib/xpu/rpp/
总用量 3784
lrwxrwxrwx 1 root root      16 7月  29 16:36 librpp_perf.so -> librpp_perf.so.1
lrwxrwxrwx 1 root root      20 7月  29 16:36 librpp_perf.so.1 -> librpp_perf.so.1.0.0
-rw-r--r-- 1 root root   63440 7月  29 16:36 librpp_perf.so.1.0.0
lrwxrwxrwx 1 root root      12 7月  29 16:41 liburpp.so -> liburpp.so.2
lrwxrwxrwx 1 root root      19 7月  29 16:41 liburpp.so.2 -> liburpp.so.2.0.21.5
-rw-r--r-- 1 root root 3807496 7月  29 16:41 liburpp.so.2.0.21.5
gytest@gytest-pc:~/Yan_debug/0729_release$ 
```

**包卸载操作：**

```bash
#查询是否有关联包
dpkg -l | grep -Ei "rpp|azurengine|xdl"

kos@kos-pc:~/Yan_debug$ dpkg -l | grep -Ei "rpp|azurengine|xdl"
ii  xdl-rpp-dkms                                  1.0.0                                       all          XDL RPP dkms package
ii  xdl-rpp-runtime                               1.0.1                                       amd64        XDL RPP runtime package
kos@kos-pc:~/Yan_debug$ 

# 卸载（彻底）
sudo dpkg --purge xdl-rpp-runtime
sudo dpkg --purge xdl-rpp-dkms


***xdl-rpp-dkms包卸载前后检查工作：***
```


卸载操作日志：

```
gytest@gytest-pc:~/Yan_debug$ sudo dpkg --purge xdl-rpp-dkms
(正在读取数据库 ... 系统当前共安装有 251490 个文件和目录。)
正在卸载 xdl-rpp-dkms (1-1) ...

-------- Uninstall Beginning --------
Module:  rpp-dkms
Version: 2.0.16.6
Kernel:  5.4.18-155-generic (aarch64)
-------------------------------------

Status: Before uninstall, this module version was ACTIVE on this kernel.

rpp.ko:
 - Uninstallation
   - Deleting from: /lib/modules/5.4.18-155-generic/kernel/extra/
 - Original module
   - No original module was found for this module on this kernel.
   - Use the dkms install command to reinstall any previous module version.

depmod...

DKMS: uninstall completed.

------------------------------
Deleting module version: 2.0.16.6
completely from the DKMS tree.
------------------------------
Done.
正在清除 xdl-rpp-dkms (1-1) 的配置文件 ...
gytest@gytest-pc:~/Yan_debug$ 
```

卸载后确认：

```
gytest@gytest-pc:~/Yan_debug/0729_release$ ls -ll /usr/src/rpp-dkms-*/ |head
ls: 无法访问'/usr/src/rpp-dkms-*/': 没有那个文件或目录
gytest@gytest-pc:~/Yan_debug/0729_release$ 
```

***xdl-rpp-runtime包卸载前后检查工作：***

卸载操作日志：

```bash
gytest@gytest-pc:~/Yan_debug/0729_release$ sudo dpkg --purge xdl-rpp-runtime
(正在读取数据库 ... 系统当前共安装有 251386 个文件和目录。)
正在卸载 xdl-rpp-runtime (1.0.0) ...
gytest@gytest-pc:~/Yan_debug/0729_release$ 
```

卸载后确认：

```bash
gytest@gytest-pc:~/Yan_debug$ ls -ll /etc/ld.so.conf.d/rpp_Kylin.conf
ls: 无法访问'/etc/ld.so.conf.d/rpp_Kylin.conf': 没有那个文件或目录
gytest@gytest-pc:~/Yan_debug$ 
```


**Llama.cpp应用编译说明：**

编译环境依赖xdl-rpp-runtime-dev_1.0.0_all.deb:

```bash
sudo dpkg -i xdl-rpp-runtime-dev_1.0.0_all.deb
```

切换到build目录，拷贝以下命令后，敲回车执行cmake：

```bash
cmake .. \
	  -DGGML_RPP=ON \
	  -DGGML_RPP_USE_UBATCH=ON \
	  -DCMAKE_BUILD_TYPE=Debug \
	  -DGGML_RPP_USE_BF16=ON \
	  -DLLAMA_CURL=OFF \
	  -DGGML_RPP_USE_DFS=ON \
	  -DRPP_INCLUDE_DIR=/usr/include/xpu/rpp \
	  -DRPP_DRV_LIBRARY=/usr/lib/xpu/rpp/liburpp.so
```

D3000 编译问题说明：

```bash
cmake .. \
	  -DGGML_RPP=ON \
	  -DGGML_RPP_USE_UBATCH=ON \
	  -DCMAKE_BUILD_TYPE=Debug \
	  -DGGML_RPP_USE_BF16=ON \
	  -DLLAMA_CURL=OFF \
	  -DGGML_RPP_USE_DFS=ON \
      -DGGML_CPU_NATIVE=OFF \
      -DGGML_NATIVE=OFF \
      -DGGML_CPU_ARM_ARCH=armv8-a \
      -DGGML_CPU_LLAMAFILE=OFF \
	  -DRPP_INCLUDE_DIR=/usr/include/xpu/rpp \
	  -DRPP_DRV_LIBRARY=/usr/lib/xpu/rpp/liburpp.so
```


执行编译：

```bash
make -j8
```

编译成功后，查询依赖：

```bash
ldd ./bin/llama-simple-chat 
```

编译成功后卸载

```bash
sudo dpkg --purge xdl-rpp-runtime-dev
```

查询依赖日志信息：

<details>
<summary>展开/收起日志</summary>

```
kos@kos-pc:~/workspace/test_framwork/test_llm_cpp/llama.cpp/build$ ldd ./bin/llama-simple-chat 
        linux-vdso.so.1 (0x00007ffc787fc000)
        libllama.so.0 => /home/kos/workspace/test_framwork/test_llm_cpp/llama.cpp/build/bin/libllama.so.0 (0x00007f0b0a6d7000)
        libggml.so.0 => /home/kos/workspace/test_framwork/test_llm_cpp/llama.cpp/build/bin/libggml.so.0 (0x00007f0b0a6a5000)
        libggml-cpu.so.0 => /home/kos/workspace/test_framwork/test_llm_cpp/llama.cpp/build/bin/libggml-cpu.so.0 (0x00007f0b0a448000)
        libggml-rpp.so.0 => /home/kos/workspace/test_framwork/test_llm_cpp/llama.cpp/build/bin/libggml-rpp.so.0 (0x00007f0b09dcb000)
        libggml-base.so.0 => /home/kos/workspace/test_framwork/test_llm_cpp/llama.cpp/build/bin/libggml-base.so.0 (0x00007f0b09c2d000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f0b09a18000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f0b099fd000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f0b099da000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f0b097f3000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f0b096a4000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f0b0b0ff000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f0b0969e000)
        libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007f0b0965a000)
        liburpp.so.2 => /usr/lib/xpu/rpp/liburpp.so.2 (0x00007f0b08b9d000)
        librpp_perf.so.1 => /usr/lib/xpu/rpp/librpp_perf.so.1 (0x00007f0afbded000)
```

</details>
