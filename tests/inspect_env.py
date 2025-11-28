import sys, site, pkgutil, subprocess
print('python', sys.version)
print('sys.path:')
for p in sys.path:
    print('  ', p)
print('\nuser site:', site.getusersitepackages() if hasattr(site, 'getusersitepackages') else 'N/A')
print('\nFirst 40 installed packages (pip list):')
subprocess.run(['/root/.local/bin/pip', 'list', '--format=columns'], check=False)
print('\nCheck tensorflow loader:', pkgutil.find_loader('tensorflow'))
