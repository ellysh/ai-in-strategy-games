#!/bin/bash

echo "## Версия" > version.md
echo -n "# " >> version.md
git show --no-patch --format=%ci HEAD >> version.md
echo -n "# " >> version.md
git rev-parse HEAD >> version.md
