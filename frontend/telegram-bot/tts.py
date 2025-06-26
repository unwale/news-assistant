import asyncio
import json
import ssl
import time
from typing import Optional
from uuid import uuid4

import aiohttp


class SberSpeechAPI:
    def __init__(self, auth_key: str):
        self.auth_key = auth_key
        self.access_token: Optional[str] = None
        self.token_expiry: float = 0
        self.oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.synthesize_url = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
        self.session: Optional[aiohttp.ClientSession] = None
        self.ssl_context = ssl.create_default_context(
            cafile="/certificates/rootca_ssl_rsa2022.pem"
        )

    async def start_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_access_token(self) -> str:
        current_time = time.time()
        if self.access_token and current_time < self.token_expiry:
            return self.access_token

        await self.start_session()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid4()),
            "Authorization": f"Basic {self.auth_key}",
        }
        payload = {"scope": "SALUTE_SPEECH_PERS"}

        try:
            async with self.session.post(
                self.oauth_url, headers=headers, data=payload, ssl=self.ssl_context
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get access token: {response.status}")

                data = await response.json()
                self.access_token = data.get("access_token")
                print(self.access_token)
                self.token_expiry = current_time + 29 * 60
                return self.access_token
        except Exception as e:
            raise Exception(f"Error fetching access token: {str(e)}")

    async def synthesize_text(self, text: str, format: str = "opus") -> dict:
        await self.start_session()
        await self.get_access_token()

        headers = {
            "Content-Type": "application/ssml",
            "Authorization": f"Bearer {self.access_token}",
        }
        params = {"format": format, "voice": "Nec_24000"}

        try:
            async with self.session.post(
                self.synthesize_url,
                headers=headers,
                params=params,
                data=text.encode("utf-8"),
                ssl=self.ssl_context,
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to synthesize text: {response}")

                return await response.read()
        except Exception as e:
            raise Exception(f"Error synthesizing text: {str(e)}")

    async def periodic_token_refresh(self):
        while True:
            try:
                await self.get_access_token()
            except Exception as e:
                raise Exception(f"Error refreshing Salute Speech token: {str(e)}")
            await asyncio.sleep(29 * 60)


if __name__ == "__main__":

    async def main():
        api = SberSpeechAPI(
            "MDg1YWViNDctZmQzYi00NTZlLWE2ZjktYmJmMmU2MzdmMjFjOjBkMDM3NTE5LWNmMTUtNDU3MS1hNGUyLTUwMWQwMTAwNWUyYQ=="
        )
        print(await api.synthesize_text("Привет! Меня зовут Никита."))
        api.close_session()

    asyncio.run(main())
