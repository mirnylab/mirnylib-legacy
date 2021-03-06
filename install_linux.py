'''Add the parent directory ../ to one of the bash settings files.
'''
import os

libpath = os.getcwd()
libpath = os.path.normpath(libpath.replace(os.path.expanduser('~'), '$HOME/'))

export_line = 'export PYTHONPATH="$PYTHONPATH:{0}"'.format(libpath)

profiles = [os.path.expanduser(i) for i in ['~/.bash_profile', '~/.bashrc']]

# Do nothing if the library is already exported.
for profile_path in profiles:
    setflag = 1
    if os.path.isfile(profile_path):
        for line in open(profile_path):
            if export_line in line:
                print 'The PYTHONPATH is already set in {0}'.format(
                    profile_path)
                setflag = 0
                break

    if setflag == 1:

        profile_file = open(profile_path, 'a')
        profile_file.writelines(
            ['\n# Added by the mirnylab install script.\n',
             export_line,
             '\n'])
        print 'PYTHONPATH is added to {0}'.format(profile_path)

os.chdir("mirnylib")
for name in ["numutils_new.so", "numutils_new.c", "numutils_new.cpp"]:
    if os.path.exists(name):
        os.remove(name)
os.chdir("..")
print "Building cython modules.."
print "If it does not work, check cython version (need 0.16+)"
os.system("python setup.py build_ext --inplace")
