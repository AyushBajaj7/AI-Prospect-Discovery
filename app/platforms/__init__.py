from app.platforms.google import GoogleAdapter
from app.platforms.linkedin import LinkedInAdapter
from app.platforms.x_platform import XAdapter
from app.platforms.youtube import YouTubeAdapter
from app.platforms.instagram import InstagramAdapter
from app.platforms.facebook import FacebookAdapter
from app.platforms.threads import ThreadsAdapter
from app.platforms.whatsapp import WhatsAppAdapter

ALL_PLATFORMS = [
    LinkedInAdapter(),
    XAdapter(),
    YouTubeAdapter(),
    InstagramAdapter(),
    FacebookAdapter(),
    ThreadsAdapter(),
    WhatsAppAdapter(),
    GoogleAdapter(),
]
