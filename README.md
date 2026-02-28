# NUWA-TRANSLATOR-RealTime


نظام ترجمة صوتية فورية من الإنجليزي للعربي يعمل على أي صوت يطلع من جهازك (ميكروفون أو صوت النظام) ويعرض الترجمة لحظياً على شاشة صغيرة Overlay.

طريقة الاستخدام

تثبيت المكتبات المطلوبة:

pip install soundcard numpy torch faster-whisper transformers tkinter

تشغيل البرنامج:

python nuwa_translator.py

واجهة التطبيق:

اضغط Start Translation ▶ لبدء التقاط الصوت والترجمة.

الترجمة الإنجليزية تظهر أولاً، ثم العربية مباشرة.

الترجمة تظهر على شاشة صغيرة أسفل الشاشة، وتختفي بعد 4 ثواني من الصمت.

لإيقاف الترجمة، اضغط Stop Translation ⏹.

ملاحظات

الإصدار الأول فقط، قريباً سيتم إضافة تحسينات وتجربة أفضل.

يعتمد على:

Faster Whisper لتحويل الصوت لنص.

Hugging Face Transformers للترجمة.

Soundcard لالتقاط الصوت.

Tkinter للواجهة والـ Overlay.

لتفاصيل اكثر
https://www.linkedin.com/posts/dhiyaa-aldeen_%D8%A7%D9%84%D8%A5%D8%B5%D8%AF%D8%A7%D8%B1-%D8%A7%D9%84%D8%A3%D9%88%D9%84-%D9%85%D9%86-nuwa-translator-real-activity-7433599071162044416-vaZX?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFKprzYB2SSEoMJynpf3vRDbUlXR48O_fow
