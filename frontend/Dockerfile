# deps: install node_modules in its own layer, cached on package*.json.
FROM node:22-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

# builder: produce the standalone server output (next.config.ts sets
# output: "standalone").
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* vars are inlined into the client bundle at build time, and
# every fetching page in this app is a client component -- so this must be
# the URL the BROWSER can reach (http://localhost:8000), never a
# docker-compose service name like http://api:8000.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# runner: minimal production image -- just the standalone server output.
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

USER node

ENV PORT=3000 HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
