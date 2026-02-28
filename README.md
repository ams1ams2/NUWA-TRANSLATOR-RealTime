# NUWA-TRANSLATOR-RealTime

NUWA TRANSLATOR REAL TIME

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
