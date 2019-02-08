import re, sys, os
import subprocess
import json
from random import randint


class GetData:

    def split_lines(self, s):
        """Splits lines in a way that works even on Windows and old devices.
        Windows will see \r\n instead of \n, old devices do the same, old devices
        on Windows will see \r\r\n.
        """
        # rstrip is used here to workaround a difference between splineslines and
        # re.split:
        # >>> 'foo\n'.splitlines()
        # ['foo']
        # >>> re.split(r'\n', 'foo\n')
        # ['foo', '']
        return re.split(r'[\r\n]+', s.rstrip())

    def get_devices(self, adb_path='adb'):
        with open(os.devnull, 'wb') as devnull:
            subprocess.check_call([adb_path, 'start-server'], stdout=devnull,
                                  stderr=devnull)
        out = self.split_lines(
            subprocess.check_output([adb_path, 'devices']).decode('utf-8'))
        # The first line of `adb devices` just says "List of attached devices", so
        # skip that.
        devices = []
        for line in out[1:]:
            if not line.strip():
                continue
            if 'offline' in line:
                continue
            serial, _ = re.split(r'\s+', line, maxsplit=1)
            devices.append(serial)
        return devices

    def getmeminfo(self, pname):
        line_dict = {}
        ls = subprocess.Popen(["adb", "shell", "dumpsys", "meminfo", pname, " | grep -E", "'Native Heap|.so'"],
                              stdout=subprocess.PIPE)
        out = ls.stdout.readlines()
        for line in out:
            if re.search("Native Heap", line):
                line = line.strip("\n")
                line_array = line.split(" ")
                line_array = " ".join(line_array).split()
                line_dict["Native_Pss"] = int(line_array[2])
                line_dict["Native_Private_Dirty"] = int(line_array[3])
                line_dict["Native_Heap_Alloc"] = int(line_array[-2])
                line_dict["Native_Heap_Free"] = int(line_array[-1])

                for line_sub in out:
                    print line_sub
                    if re.search(".so", line_sub):
                        line_sub = line_sub.strip("\n")
                        line_array = line_sub.split(" ")
                        line_array = " ".join(line_array).split()
                        line_dict["code_Pss"] = int(line_array[2])
                        line_dict["code_Private_Dirty"] = int(line_array[3])
                        break
            break

        if line_dict:
            #return line_dict["Total_Heap_Alloc"]
            return line_dict
        return "App not started.Unable to fetch meminfo...."

    def getcpuinfo(self, pname):
        lm = subprocess.Popen(["adb", "shell", "dumpsys", "cpuinfo", "| grep -i", pname], stdout=subprocess.PIPE)
        out = lm.stdout.readlines()
        for line in out:
            return float(line.strip("\n").split("%")[0])
        else:
            return "App not started.Unable to fetch cpuinfo"

    def getcpucores(self):
        line_dict_cpu = {}
        try:
            for i in range(0, 4):
                command = "adb shell cat sys/devices/system/cpu/cpu" + str(i) + "/cpufreq/scaling_cur_freq"
                lo = os.popen(command)
                out = int(lo.readlines()[0].strip('\n'))
                line_dict_cpu["CPU" + str(i)] = out
            return json.dumps(line_dict_cpu)
        except OSError:
            return "Unable to fetch CPU core details"

        if line_dict_cpu:
            return json.dumps(line_dict_cpu)
        return "App not started.Unable to fetch CPU Core"

    def getJenkyFrames(self):
        jenky_frames_output = os.popen("adb shell dumpsys gfxinfo")
        for count, line in enumerate(jenky_frames_output.readlines()):
            if 'Janky frame' in line:
                return str(line.split()[-2])

    def getbatterystats(self):
        try:
            out = os.popen("adb shell dumpsys batterystats")
            return out.readlines()[0]
        except OSError:
            return "Unable to fetch Battery Stats"

    def getcurrentevent(self):
        try:
            p = subprocess.Popen(['adb', 'logcat', '-d'], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            s = p.stdout.readlines()
            mylist = []
            for line in s:
                if 'cmp' in line:
                    s1 = line.split('cmp')[1]
                    mylist.append(s1[1:])

            return mylist[-1]
        except OSError:
            return "Unable to fetch current event"

    def getframestats(self, pname):
        mylist1 = []
        vsync = []
        FrameCompleted = []
        pdata = 0
        framestats_output = os.popen("adb shell dumpsys gfxinfo " + pname + " framestats")
        for count, line in enumerate(framestats_output.readlines()):

            if '---PROFILEDATA---' in line:
                pdata += 1
                continue

            if pdata == 1:
                mylist1.append(line)

            if pdata == 2:
                for i in mylist1:
                    framestats = i.split(',')
                    vsync.append(framestats[2])
                    FrameCompleted.append(framestats[13])
                return zip(vsync, FrameCompleted)

    def test_rand(self):
        return randint(0, 9)


#print GetData().getmeminfo("com.google.android.perftesting")
