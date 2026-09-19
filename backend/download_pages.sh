#!/bin/bash
mkdir -p pages

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Louvre" -o "pages/page1.html"
echo "Downloaded page1.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Australia" -o "pages/page2.html"
echo "Downloaded page2.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/IPhone" -o "pages/page3.html"
echo "Downloaded page3.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Apple" -o "pages/page4.html"
echo "Downloaded page4.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Coastal_erosion" -o "pages/page5.html"
echo "Downloaded page5.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Photosynthesis" -o "pages/page6.html"
echo "Downloaded page6.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Eiffel_Tower" -o "pages/page7.html"
echo "Downloaded page7.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Mona_Lisa" -o "pages/page8.html"
echo "Downloaded page8.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Bread" -o "pages/page9.html"
echo "Downloaded page9.html"
sleep 1

curl -sL -A "Mozilla/5.0" "https://en.wikipedia.org/wiki/Internal_combustion_engine" -o "pages/page10.html"
echo "Downloaded page10.html"

echo "All downloads complete."
