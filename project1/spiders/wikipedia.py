import scrapy


class WikipediaSpider(scrapy.Spider):
    name = "wikipedia"
    allowed_domains = ["en.wikipedia.org"]
    start_urls = ["https://en.wikipedia.org/wiki/Albert_Camus"]
    
    custom_settings = {
        'USER_AGENT': 'Olek-and-Karol',
        'DEPTH_LIMIT': 2,
    }

    def parse(self, response):
        content = response.css("#mw-content-text .mw-parser-output")

        # Exclude tables, references, navboxes
        for element in content.css('table, .reflist, .navbox, .mw-editsection'):
            element.drop()

        text = content.css("p::text, p a::text").getall()
        text = " ".join(t.strip() for t in text if t.strip())

        yield {
            "url": response.url,
            "text": text
        }

        for href in response.css(".mw-parser-output a::attr(href)").getall():
            if href.startswith("/wiki/") and ":" not in href:
                yield response.follow(href, self.parse)
