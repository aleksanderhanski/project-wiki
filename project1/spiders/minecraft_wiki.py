import scrapy


class MinecraftWikiSpider(scrapy.Spider):
    name = "minecraft-wiki"
    allowed_domains = ["minecraft.fandom.com"]
    start_urls = ["https://minecraft.fandom.com/wiki/Gameplay"]

    def parse(self, response):
        content = response.css("#mw-content-text .mw-parser-output")

        text = content.css("::text").getall()
        text = " ".join(t.strip() for t in text if t.strip())

        yield {
            "url": response.url,
            "text": text
        }

        for href in response.css(".mw-parser-output a::attr(href)").getall():
            if href.startswith("/wiki/") and ":" not in href:
                yield response.follow(href, self.parse)
