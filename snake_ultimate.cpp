// This literally just crashes on boot after compilation... At least for me. 

#include "raylib.h"
#include <vector>
#include <deque>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <iostream>

const int GRID_SIZE = 40;

struct Point {
    int x, y, z;
};

bool IsInside(Point p) {
    return p.x >= 0 && p.x < GRID_SIZE && p.z >= 0 && p.z < GRID_SIZE;
}

float GetDistance(Point a, Point b) {
    return sqrtf((float)(powf(a.x - b.x, 2) + powf(a.z - b.z, 2)));
}

// Moved from Lambda to Global Function for maximum compatibility
void DrawSnake(std::deque<Point>& s, Color baseColor, const std::vector<Point>& apples) {
    if (s.empty()) return;

    for (auto& p : s) {
        float totalIntensity = 0.0f;
        for (auto& a : apples) {
            float dist = GetDistance(p, a);
            totalIntensity += (25.0f / (dist + 4.0f));
        }
        float brightness = fminf(1.0f, totalIntensity * 0.8f);
        Color finalCol = { (unsigned char)(baseColor.r * brightness),
            (unsigned char)(baseColor.g * brightness),
            (unsigned char)(baseColor.b * brightness), 255 };
            DrawCube((Vector3){(float)p.x, 1.0f, (float)p.z}, 0.9f, 0.9f, 0.9f, finalCol);
    }
}

int main() {
    const int screenWidth = 800;
    const int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "3D Snake - Stable Edition");

    // Camera Variables
    float camAngle = 0.0f;
    float camDistance = 60.0f;
    float camHeight = 45.0f;

    Camera3D camera = { 0 };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    // Game State - Guaranteed Initial Population
    std::deque<Point> playerSnake;
    playerSnake.push_back({20, 1, 20});
    playerSnake.push_back({19, 1, 20});
    playerSnake.push_back({18, 1, 20});

    std::deque<Point> npcSnake;
    npcSnake.push_back({5, 1, 5});
    npcSnake.push_back({6, 1, 5});
    npcSnake.push_back({7, 1, 5});

    std::vector<Point> apples = {{10, 1, 10}, {30, 1, 30}};

    Point pDir = {1, 0, 0};
    int moveTimer = 0;
    int moveInterval = 12;

    srand(time(NULL));
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        // --- CAMERA CONTROLS ---
        if (IsKeyDown(KEY_RIGHT)) camAngle += 1.5f;
        if (IsKeyDown(KEY_LEFT))  camAngle -= 1.5f;
        if (IsKeyDown(KEY_UP))    camDistance = fmaxf(20.0f, camDistance - 0.8f);
        if (IsKeyDown(KEY_DOWN))  camDistance = fminf(100.0f, camDistance + 0.8f);

        // Calculate and Sanitize Camera Position
        float posX = (GRID_SIZE / 2.0f) + cosf(camAngle * DEG2RAD) * camDistance;
        float posZ = (GRID_SIZE / 2.0f) + sinf(camAngle * DEG2RAD) * camDistance;

        // Safety check: If math produces NaN, reset to default to prevent crash
        if (std::isnan(posX) || std::isnan(posZ)) {
            posX = (GRID_SIZE / 2.0f);
            posZ = (GRID_SIZE / 2.0f);
        }

        camera.position = (Vector3){ posX, camHeight, posZ };
        camera.target = (Vector3){ (float)GRID_SIZE/2.0f, 1.0f, (float)GRID_SIZE/2.0f };
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };

        // --- PLAYER INPUT ---
        if (IsKeyPressed(KEY_W)) pDir = {0, 0, -1};
        if (IsKeyPressed(KEY_S)) pDir = {0, 0, 1};
        if (IsKeyPressed(KEY_A)) pDir = {-1, 0, 0};
        if (IsKeyPressed(KEY_D)) pDir = {1, 0, 0};

        // --- NPC PATHFINDING & BEHAVIOR ---
        Point bestMove = {0, 1, 0};
        bool npcHasValidMove = false;

        if (!npcSnake.empty() && !apples.empty()) {
            Point nearestApple = apples[0];
            float minDist = 999.0f;
            for(auto& a : apples) {
                float d = GetDistance(npcSnake.front(), a);
                if (d < minDist) { minDist = d; nearestApple = a; }
            }

            Point directions[4] = {{1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1}};
            float bestDist = 999.0f;
            for(int i=0; i<4; i++) {
                Point testMove = {npcSnake.front().x + directions[i].x, 1, npcSnake.front().z + directions[i].z};
                if (IsInside(testMove)) {
                    bool collision = false;
                    for(auto& p : npcSnake) if(p.x == testMove.x && p.z == testMove.z) collision = true;
                    for(auto& p : playerSnake) if(p.x == testMove.x && p.z == testMove.z) collision = true;

                    if (!collision) {
                        float d = GetDistance(testMove, nearestApple);
                        if (d < bestDist) {
                            bestDist = d;
                            bestMove = testMove;
                            npcHasValidMove = true;
                        }
                    }
                }
            }
        }

        // --- MOVEMENT & COLLISION LOGIC ---
        moveTimer++;
        if (moveTimer >= moveInterval) {
            moveTimer = 0;

            // Move Player
            if (!playerSnake.empty()) {
                Point nextP = {playerSnake.front().x + pDir.x, 1, playerSnake.front().z + pDir.z};
                bool pCollision = false;
                if (!IsInside(nextP)) pCollision = true;
                else {
                    for(auto& p : playerSnake) if(p.x == nextP.x && p.z == nextP.z) pCollision = true;
                    for(auto& p : npcSnake) if(p.x == nextP.x && p.z == nextP.z) pCollision = true;
                }

                if (pCollision) {
                    playerSnake.clear();
                    playerSnake.push_back({20, 1, 20});
                } else {
                    playerSnake.push_front(nextP);
                    bool ateApple = false;
                    for (auto& a : apples) {
                        if (nextP.x == a.x && nextP.z == a.z) {
                            a.x = rand() % GRID_SIZE;
                            a.z = rand() % GRID_SIZE;
                            ateApple = true;
                        }
                    }
                    if (!ateApple) playerSnake.pop_back();
                }
            }

            // Move NPC
            if (!npcSnake.empty()) {
                if (npcHasValidMove) {
                    npcSnake.push_front(bestMove);
                    npcSnake.pop_back();
                } else {
                    Point nextN = {npcSnake.front().x + 1, 1, npcSnake.front().z};
                    if (IsInside(nextN)) npcSnake.push_front(nextN);
                    else npcSnake.pop_back();
                }

                for (auto& a : apples) {
                    if (npcSnake.front().x == a.x && npcSnake.front().z == a.z) {
                        a.x = rand() % GRID_SIZE;
                        a.z = rand() % GRID_SIZE;
                    } else {
                        npcSnake.pop_back();
                    }
                }
            }
        }

        // --- RENDERING ---
        BeginDrawing();
        ClearBackground(BLACK);
        BeginMode3D(camera);

        for (int x = 0; x < GRID_SIZE; x++) {
            for (int z = 0; z < GRID_SIZE; z++) {
                DrawCube((Vector3){(float)x, 0.0f, (float)z}, 1.0f, 0.1f, 1.0f, DARKGRAY);
            }
        }

        for (auto& a : apples) DrawCube((Vector3){(float)a.x, 1.0f, (float)a.z}, 0.8f, 0.8f, 0.8f, RED);

        DrawSnake(playerSnake, GREEN, apples);
        DrawSnake(npcSnake, BLUE, apples);

        EndMode3D();

        DrawText("Arrows: Rotate/Zoom | WASD: Move Snake", 10, 10, 20, WHITE);
        DrawFPS(10, 40);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
