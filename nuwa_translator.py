import soundcard as sc
import numpy as np
import torch
import time
import threading
import queue
import re
import tkinter as tk
from tkinter import ttk
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# 1. إعدادات النظام (SETTINGS)
# ==========================================
APP_NAME = "NUWA TRANSLATOR REAL TIME"
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5      
WHISPER_MODEL = "base"    # يمكن رفعه إلى "small" أو "medium" لدقة أعلى إذا كان جهازك قوياً
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-en-ar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# أيقونة مدمجة (كرة أرضية/ترجمة) لضمان ظهورها في شريط المهام
ICON_BASE64 = """
R0lGODlhIAAgAPEBAAAAAP///wAAAAAAACH5BAEAAAIALAAAAAAgACAAAAKwhI+py+0Po5y02ouz
3rz7D4biSJbmiabqyrbuC8fyTNf2jef6zvf+DwwKhcQiccksEgoAowECAQEAOw==
"""

# قاموس الأصوات والموسيقى (يتعرف على ما يخرجه Whisper)
SOUND_EVENTS = {
    "music": "🎵 [موسيقى] 🎵",
    "applause": "👏 [تصفيق] 👏",
    "laughter": "😂 [ضحك] 😂",
    "laughs": "😂 [ضحك] 😂",
    "sighs": "💨 [تنهيدة]",
    "cheers": "🎉 [هتاف] 🎉",
    "clears throat": "🗣️ [نحنحة]",
    "bell": "🔔 [صوت جرس] 🔔"
}

# ==========================================
# 2. الفئة الرئيسية للتطبيق (MAIN APP CLASS)
# ==========================================
class NuwaTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("450x220")
        self.root.resizable(False, False)
        self.root.configure(bg="#1A1A2E")
        
        # إعداد الأيقونة لتظهر في النافذة وشريط المهام
        try:
            self.icon_image = tk.PhotoImage(data=ICON_BASE64)
            self.root.iconphoto(True, self.icon_image)
        except Exception as e:
            print("لم يتم تحميل الأيقونة:", e)

        # المتغيرات والحالة
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=15)
        self.text_queue = queue.Queue(maxsize=15)
        
        # سياق تفريغ الصوت (لتحسين جودة الجمل المترابطة)
        self.transcription_context = "Translate accurately, keeping standard punctuation and grammar. "
        
        # تحميل النماذج
        self.whisper_model = None
        self.tokenizer = None
        self.translator = None
        self.overlay = None
        self.last_speech_time = time.time()

        self.setup_control_ui()

    def setup_control_ui(self):
        """بناء واجهة لوحة التحكم الرئيسية بطابع حديث"""
        title = tk.Label(self.root, text=APP_NAME, font=("Segoe UI", 18, "bold"), fg="#0F3460", bg="#1A1A2E")
        title.config(fg="#E94560") # لون مميز للعنوان
        title.pack(pady=15)

        self.status_label = tk.Label(self.root, text="Ready to start.", font=("Segoe UI", 11), fg="#A2A2BD", bg="#1A1A2E")
        self.status_label.pack(pady=5)

        self.start_btn = tk.Button(self.root, text="▶ Start Translation", font=("Segoe UI", 12, "bold"), 
                                   bg="#0F3460", fg="white", activebackground="#E94560", 
                                   activeforeground="white", relief="flat", cursor="hand2", 
                                   command=self.toggle_translation)
        self.start_btn.pack(pady=15, ipadx=30, ipady=8)

    def toggle_translation(self):
        """تشغيل / إيقاف المعالجة"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(text="⏹ Stop Translation", bg="#E94560", activebackground="#B71C1C")
            self.status_label.config(text="Loading AI Models... Please wait ⏳")
            self.root.update()
            
            threading.Thread(target=self.init_system, daemon=True).start()
        else:
            self.is_running = False
            self.start_btn.config(text="▶ Start Translation", bg="#0F3460", activebackground="#0F3460")
            self.status_label.config(text="Translation Stopped.")
            if self.overlay:
                self.overlay.destroy()
                self.overlay = None

    def init_system(self):
        """تحميل النماذج وفتح الـ Overlay"""
        if self.whisper_model is None:
            self.whisper_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=8)
            self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL)
            self.translator = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL).to(DEVICE)
        
        self.status_label.config(text=f"System Active 🟢 | Device: {DEVICE.upper()}")
        self.root.after(0, self.create_overlay)
        
        threading.Thread(target=self.audio_listener, daemon=True).start()
        threading.Thread(target=self.transcribe_worker, daemon=True).start()
        threading.Thread(target=self.translate_worker, daemon=True).start()
        threading.Thread(target=self.watchdog, daemon=True).start()

    def create_overlay(self):
        """إنشاء الشاشة المتراكبة (Overlay) بشكل أنيق"""
        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.90)
        self.overlay.configure(bg="#000000")

        self.en_var = tk.StringVar()
        self.ar_var = tk.StringVar()

        # الإطار الرئيسي للنص
        frame = tk.Frame(self.overlay, bg="#000000")
        frame.pack(padx=20, pady=10)

        # النص الإنجليزي (رمادي مائل)
        self.en_label = tk.Label(frame, textvariable=self.en_var, font=("Segoe UI", 14, "italic"),
                            fg="#B0B0B0", bg="#000000", wraplength=1200, justify="center")
        self.en_label.pack(pady=(0, 5))

        # النص العربي (أكبر، أبيض ناصع مع لمسة احترافية)
        self.ar_label = tk.Label(frame, textvariable=self.ar_var, font=("Tajawal", 24, "bold"),
                            fg="#FFFFFF", bg="#000000", wraplength=1200, justify="center")
        self.ar_label.pack()

        self.update_overlay_position()

    def update_overlay_position(self):
        """تحديث موقع الشاشة المتراكبة لتكون أسفل الشاشة دائماً"""
        if not self.overlay or not self.overlay.winfo_exists(): return
        self.overlay.update_idletasks()
        screen_w = self.overlay.winfo_screenwidth()
        screen_h = self.overlay.winfo_screenheight()
        w = self.overlay.winfo_reqwidth()
        h = self.overlay.winfo_reqheight()
        x = (screen_w - w) // 2
        y = screen_h - h - 80 # مسافة من الأسفل
        self.overlay.geometry(f"+{x}+{y}")

    # ==========================================
    # 3. معالجة الصوت والأحداث
    # ==========================================
    def check_for_sound_events(self, text):
        """البحث عن وصف الأصوات داخل النص مثل (music) أو [applause]"""
        # استخراج ما بين الأقواس
        tags = re.findall(r'[\(\[](.*?)[\)\]]', text.lower())
        for tag in tags:
            for key, emoji_text in SOUND_EVENTS.items():
                if key in tag:
                    return emoji_text
        return None

    def audio_listener(self):
        """التقاط الصوت من النظام (Loopback)"""
        speaker = sc.default_speaker()
        mics = sc.all_microphones(include_loopback=True)
        loopback = next((m for m in mics if speaker.name in m.name), mics[0]) 

        chunk_frames = int(CHUNK_DURATION * SAMPLE_RATE)
        
        with loopback.recorder(samplerate=SAMPLE_RATE) as rec:
            while self.is_running:
                frames = rec.record(numframes=chunk_frames)
                if frames.ndim > 1:
                    frames = frames.mean(axis=1)
                
                audio_data = frames.astype(np.float32)
                volume_norm = np.linalg.norm(audio_data) * 10
                
                if volume_norm > 1.5: 
                    if not self.audio_queue.full():
                        self.audio_queue.put(audio_data)
                else:
                    time.sleep(0.05)

    def transcribe_worker(self):
        """تحويل الصوت إلى نص إنجليزي مع حقن السياق"""
        while self.is_running:
            try:
                audio = self.audio_queue.get(timeout=1)
                # حقن السياق السابق هنا
                segments, _ = self.whisper_model.transcribe(
                    audio, language="en", task="transcribe", 
                    beam_size=1, temperature=0.0, vad_filter=True,
                    initial_prompt=self.transcription_context
                )
                
                text = " ".join([s.text.strip() for s in segments]).strip()
                
                if text:
                    self.last_speech_time = time.time()
                    
                    # تحديث السياق بآخر الكلمات (نحتفظ بآخر 100 حرف لعدم إرهاق الذاكرة)
                    self.transcription_context = text[-100:] 

                    if not self.text_queue.full():
                        self.text_queue.put(text)
            except queue.Empty:
                continue
            except Exception as e:
                pass

    def translate_worker(self):
        """ترجمة النص إلى العربية والتعرف على الأصوات"""
        while self.is_running:
            try:
                en_text = self.text_queue.get(timeout=1)
                
                # 1. التحقق من وجود أحداث صوتية (موسيقى، تصفيق، الخ)
                sound_event = self.check_for_sound_events(en_text)
                
                self.root.after(0, self.en_var.set, en_text)
                
                if sound_event:
                    # إذا كان مجرد صوت، نضع الأيقونة ونتخطى الترجمة للسرعة
                    self.root.after(0, self.ar_var.set, sound_event)
                else:
                    # 2. الترجمة الفعلية
                    inputs = self.tokenizer(en_text, return_tensors="pt", padding=True).to(DEVICE)
                    gen = self.translator.generate(**inputs, max_length=128, num_beams=1, use_cache=True)
                    ar_text = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

                    self.root.after(0, self.ar_var.set, ar_text)

                self.root.after(0, self.update_overlay_position)

            except queue.Empty:
                continue
            except Exception as e:
                pass

    def watchdog(self):
        """إخفاء الترجمة بعد 4 ثواني من الصمت"""
        while self.is_running:
            time.sleep(0.5)
            if time.time() - self.last_speech_time > 4.0:
                if self.en_var.get() != "":
                    self.root.after(0, self.en_var.set, "")
                    self.root.after(0, self.ar_var.set, "")
                    self.root.after(0, self.update_overlay_position)

# ==========================================
# 4. نقطة الانطلاق (ENTRY POINT)
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = NuwaTranslatorApp(root)
    root.mainloop()