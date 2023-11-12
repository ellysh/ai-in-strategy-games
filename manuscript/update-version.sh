#!/bin/bash

echo "### Версия" > version.md
git show --no-patch --format=%ci HEAD >> version.md
git rev-parse HEAD >> version.md
